use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_eou::{EncoderCache, ParakeetEOUModel};
use ndarray::{s, Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
const FMAX: f32 = 8000.0;

/// Parakeet RealTime EOU model for streaming ASR with end-of-utterance detection.
/// Uses cache-aware streaming with audio buffering for pre-encode context.
pub struct ParakeetEOU {
    model: ParakeetEOUModel,
    tokenizer: tokenizers::Tokenizer,
    encoder_cache: EncoderCache,
    state_h: Array3<f32>,
    state_c: Array3<f32>,
    last_token: Array2<i32>,
    blank_id: i32,
    eou_id: i32,
    mel_basis: Array2<f32>,
    window: Vec<f32>,
    audio_buffer: VecDeque<f32>,
    buffer_size_samples: usize,
}

impl ParakeetEOU {
    /// Load Parakeet EOU model from path
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder.onnx, decoder_joint.onnx, and tokenizer.json
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Config(format!("Failed to load tokenizer: {e}")))?;

        let vocab_size = tokenizer.get_vocab_size(true);
        let blank_id = (vocab_size - 1) as i32;
        let blank_id = if blank_id < 1000 { 1026 } else { blank_id };
        let eou_id = tokenizer
            .token_to_id("<EOU>")
            .map(|id| id as i32)
            .unwrap_or(1024);

        let exec_config = config.unwrap_or_default();
        let model = ParakeetEOUModel::from_pretrained(path, exec_config)?;

        // Buffer size: 4 seconds of audio
        // Provides long history for feature extraction context
        // Note that, I pick those "magic numbers" by looking NeMo's ring buffer approach.
        let buffer_size_samples = SAMPLE_RATE * 4; // 4 seconds = 64000 samples

        Ok(Self {
            model,
            tokenizer,
            encoder_cache: EncoderCache::new(),
            state_h: Array3::zeros((1, 1, 640)),
            state_c: Array3::zeros((1, 1, 640)),
            last_token: Array2::from_elem((1, 1), blank_id),
            blank_id,
            eou_id,
            mel_basis: Self::create_mel_filterbank(),
            window: Self::create_window(),
            audio_buffer: VecDeque::with_capacity(buffer_size_samples),
            buffer_size_samples,
        })
    }

    /// Transcribe a chunk of audio samples.
    ///
    /// # Arguments
    /// * `chunk` - Audio chunk (typically 160ms / 2560 samples at 16kHz)
    /// * `reset_on_eou` - If true, reset decoder state when end-of-utterance is detected
    ///
    /// # Streaming Behavior
    /// Cache-aware streaming
    /// - Maintains 4-second ring buffer for feature extraction context
    /// - Extracts features from full buffer
    /// - Slices last (pre_encode_cache + new_frames) for encoder input
    /// - pre_encode_cache=9 frames, new_frames=~16, total=~25 frames to encoder
    pub fn transcribe(&mut self, chunk: &[f32], reset_on_eou: bool) -> Result<String> {
        // Add new chunk to rolling buffer
        self.audio_buffer.extend(chunk.iter().copied());

        // Trim buffer to keep only the most recent samples
        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        // Wait until buffer has minimum samples (at least 1 second for stable features)
        const MIN_BUFFER_SAMPLES: usize = SAMPLE_RATE; // 1 second
        if self.audio_buffer.len() < MIN_BUFFER_SAMPLES {
            return Ok(String::new());
        }

        // Extract features from FULL buffer (provides context for feature extraction)
        let buffer_slice: Vec<f32> = self.audio_buffer.iter().copied().collect();
        let full_features = self.extract_mel_features(&buffer_slice);
        let total_frames = full_features.shape()[2];

        // Slice to take only (pre_encode_cache + new_frames) for encoder
        // pre_encode_cache = 9 frames, new_frames = ~16 for 160ms chunk
        const PRE_ENCODE_CACHE: usize = 9;
        const FRAMES_PER_CHUNK: usize = 16;
        const SLICE_LEN: usize = PRE_ENCODE_CACHE + FRAMES_PER_CHUNK;

        let start_frame = total_frames.saturating_sub(SLICE_LEN);

        let features = full_features.slice(s![.., .., start_frame..]).to_owned();
        let time_steps = features.shape()[2];

        // Encode with cache - encoder sees full buffer context
        let (encoder_out, new_cache) =
            self.model
                .run_encoder(&features, time_steps as i64, &self.encoder_cache)?;
        self.encoder_cache = new_cache;

        let total_frames = encoder_out.shape()[2];
        if total_frames == 0 {
            return Ok(String::new());
        }

        // Process all output frames (typically 1 frame per chunk)
        let new_frames = encoder_out;

        let mut text_output = String::new();

        for t in 0..new_frames.shape()[2] {
            let current_frame = new_frames.slice(s![.., .., t..t + 1]).to_owned();
            let mut syms_added = 0;

            while syms_added < 5 {
                let (logits, new_h, new_c) = self.model.run_decoder(
                    &current_frame,
                    &self.last_token,
                    &self.state_h,
                    &self.state_c,
                )?;

                let vocab = logits.slice(s![0, 0, ..]);

                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &val) in vocab.iter().enumerate() {
                    if val.is_finite() && val > max_val {
                        max_val = val;
                        max_idx = i as i32;
                    }
                }

                if max_idx == self.blank_id || max_idx == 0 {
                    break;
                }

                if max_idx == self.eou_id {
                    if reset_on_eou {
                        self.reset_states();
                        return Ok(text_output + " [EOU]");
                    }
                    break;
                }

                if max_idx as usize >= self.tokenizer.get_vocab_size(true) {
                    break;
                }

                self.state_h = new_h;
                self.state_c = new_c;
                self.last_token.fill(max_idx);

                if let Ok(decoded) = self.tokenizer.decode(&[max_idx as u32], true) {
                    text_output.push_str(&decoded);
                }
                syms_added += 1;
            }
        }
        Ok(text_output)
    }

    fn reset_states(&mut self) {
        // Soft reset: Only reset decoder states
        // at this state, we need to keep encoder cache and audio buffer flowing for continuous context
        // self.encoder_cache = EncoderCache::new();  // DON'T reset!!!
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.last_token.fill(self.blank_id);
        // self.audio_buffer.clear();  // DON'T clear!!
    }

    fn extract_mel_features(&self, audio: &[f32]) -> Array3<f32> {
        let audio_pre = Self::apply_preemphasis(audio);
        let spec = self.stft(&audio_pre);
        let mel = self.mel_basis.dot(&spec);
        let mel_log = mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln());
        mel_log.insert_axis(ndarray::Axis(0))
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(audio.len());
        if audio.is_empty() {
            return result;
        }

        let safe_x = |x: f32| if x.is_finite() { x } else { 0.0 };

        result.push(safe_x(audio[0]));
        for i in 1..audio.len() {
            result.push(safe_x(audio[i]) - PREEMPH * safe_x(audio[i - 1]));
        }
        result
    }

    fn stft(&self, audio: &[f32]) -> Array2<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let pad_amount = N_FFT / 2;
        let mut padded_audio = vec![0.0; pad_amount];
        padded_audio.extend_from_slice(audio);
        padded_audio.extend(std::iter::repeat_n(0.0, pad_amount));

        let num_frames = 1 + (padded_audio.len().saturating_sub(WIN_LENGTH)) / HOP_LENGTH;
        let freq_bins = N_FFT / 2 + 1;
        let mut spec = Array2::zeros((freq_bins, num_frames));

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            if start + WIN_LENGTH > padded_audio.len() {
                break;
            }

            let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
            for i in 0..WIN_LENGTH {
                buffer[i] = Complex::new(padded_audio[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut buffer);
            for (i, val) in buffer.iter().take(freq_bins).enumerate() {
                let mag_sq = val.norm_sqr();
                spec[[i, frame_idx]] = if mag_sq.is_finite() { mag_sq } else { 0.0 };
            }
        }
        spec
    }

    fn create_window() -> Vec<f32> {
        (0..WIN_LENGTH)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / ((WIN_LENGTH - 1) as f32)).cos())
            .collect()
    }

    fn create_mel_filterbank() -> Array2<f32> {
        let num_freqs = N_FFT / 2 + 1;

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(FMAX);

        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32))
            .collect();

        let fft_freqs: Vec<f32> = (0..num_freqs)
            .map(|i| (SAMPLE_RATE as f32 / N_FFT as f32) * i as f32)
            .collect();

        let mut weights = Array2::zeros((N_MELS, num_freqs));

        for i in 0..N_MELS {
            let left = mel_points[i];
            let center = mel_points[i + 1];
            let right = mel_points[i + 2];
            for (j, &freq) in fft_freqs.iter().enumerate() {
                if freq >= left && freq <= center {
                    weights[[i, j]] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    weights[[i, j]] = (right - freq) / (right - center);
                }
            }
        }

        for i in 0..N_MELS {
            let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
            for j in 0..num_freqs {
                weights[[i, j]] *= enorm;
            }
        }

        weights
    }
}
