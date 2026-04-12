use crate::audio::load_audio;
use crate::config::PreprocessorConfig;
use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_unified::{ParakeetUnifiedModel, UnifiedModelConfig};
use crate::nemotron::SentencePieceVocab;
use crate::timestamps::{process_timestamps, TimestampMode};
use crate::transcriber::Transcriber;
use ndarray::{Array2, Array3};
use realfft::RealFftPlanner;
use std::f64::consts::PI;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;
const FEATURE_SIZE: usize = 128;
const HOP_LENGTH: usize = 160;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const PREEMPHASIS: f32 = 0.97;
const DECODER_LSTM_DIM: usize = 640;
const DECODER_LSTM_LAYERS: usize = 2;
const SUBSAMPLING_FACTOR: usize = 8;
const MAX_SYMBOLS_PER_STEP: usize = 10;

#[derive(Debug, Clone, Copy)]
pub struct UnifiedStreamingConfig {
    pub left_context_secs: f32,
    pub chunk_secs: f32,
    pub right_context_secs: f32,
}

impl Default for UnifiedStreamingConfig {
    fn default() -> Self {
        Self {
            left_context_secs: 5.6,
            chunk_secs: 0.56,
            right_context_secs: 0.56,
        }
    }
}

impl UnifiedStreamingConfig {
    fn frames_from_secs(secs: f32) -> usize {
        ((secs * SAMPLE_RATE as f32) / HOP_LENGTH as f32).round() as usize
    }

    pub fn validate(self) -> Result<Self> {
        let left_frames = self.left_context_frames();
        let chunk_frames = self.chunk_frames();
        let right_frames = self.right_context_frames();

        if chunk_frames == 0 {
            return Err(Error::Config(
                "Unified streaming chunk size must be greater than zero".to_string(),
            ));
        }

        for (name, frames) in [
            ("left_context_secs", left_frames),
            ("chunk_secs", chunk_frames),
            ("right_context_secs", right_frames),
        ] {
            if frames % SUBSAMPLING_FACTOR != 0 {
                return Err(Error::Config(format!(
                    "{name} must map to a mel-frame count divisible by {SUBSAMPLING_FACTOR}"
                )));
            }
        }

        Ok(self)
    }

    pub fn left_context_frames(self) -> usize {
        Self::frames_from_secs(self.left_context_secs)
    }

    pub fn chunk_frames(self) -> usize {
        Self::frames_from_secs(self.chunk_secs)
    }

    pub fn right_context_frames(self) -> usize {
        Self::frames_from_secs(self.right_context_secs)
    }

    pub fn total_window_frames(self) -> usize {
        self.left_context_frames() + self.chunk_frames() + self.right_context_frames()
    }

    pub fn left_context_samples(self) -> usize {
        self.left_context_frames() * HOP_LENGTH
    }

    pub fn chunk_samples(self) -> usize {
        self.chunk_frames() * HOP_LENGTH
    }

    pub fn right_context_samples(self) -> usize {
        self.right_context_frames() * HOP_LENGTH
    }

    pub fn total_window_samples(self) -> usize {
        self.total_window_frames() * HOP_LENGTH
    }

    pub fn chunk_encoder_frames(self) -> usize {
        self.chunk_frames() / SUBSAMPLING_FACTOR
    }

    pub fn left_context_encoder_frames(self) -> usize {
        self.left_context_frames() / SUBSAMPLING_FACTOR
    }
}

pub struct ParakeetUnified {
    model: ParakeetUnifiedModel,
    vocab: SentencePieceVocab,
    preprocessor_config: PreprocessorConfig,
    mel_filterbank: Array2<f64>,
    window: Vec<f64>,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    blank_id: usize,
    streaming_config: UnifiedStreamingConfig,
    audio_buffer: Vec<f32>,
    buffer_start_sample: usize,
    next_chunk_start_sample: usize,
    accumulated_tokens: Vec<usize>,
    accumulated_timed_tokens: Vec<TimedToken>,
}

impl ParakeetUnified {
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Self::from_pretrained_with_streaming_config(
            path,
            exec_config,
            UnifiedStreamingConfig::default(),
        )
    }

    pub fn from_pretrained_with_streaming_config<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
        streaming_config: UnifiedStreamingConfig,
    ) -> Result<Self> {
        let path = path.as_ref();
        let streaming_config = streaming_config.validate()?;

        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let blank_id = vocab.size();

        let model_config = UnifiedModelConfig {
            vocab_size: vocab.size() + 1,
            blank_id,
            decoder_lstm_dim: DECODER_LSTM_DIM,
            decoder_lstm_layers: DECODER_LSTM_LAYERS,
            subsampling_factor: SUBSAMPLING_FACTOR,
        };

        let model = ParakeetUnifiedModel::from_pretrained(
            path,
            exec_config.unwrap_or_default(),
            model_config,
        )?;

        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: FEATURE_SIZE,
            hop_length: HOP_LENGTH,
            n_fft: N_FFT,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: PREEMPHASIS,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: SAMPLE_RATE,
            win_length: WIN_LENGTH,
        };

        Ok(Self {
            model,
            vocab,
            preprocessor_config,
            mel_filterbank: Self::create_mel_filterbank(),
            window: Self::create_window(),
            state_1: Array3::zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM)),
            state_2: Array3::zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM)),
            last_token: blank_id as i32,
            blank_id,
            streaming_config,
            audio_buffer: Vec::new(),
            buffer_start_sample: 0,
            next_chunk_start_sample: 0,
            accumulated_tokens: Vec::new(),
            accumulated_timed_tokens: Vec::new(),
        })
    }

    pub fn streaming_config(&self) -> UnifiedStreamingConfig {
        self.streaming_config
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }

    pub fn reset(&mut self) {
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = self.blank_id as i32;
        self.audio_buffer.clear();
        self.buffer_start_sample = 0;
        self.next_chunk_start_sample = 0;
        self.accumulated_tokens.clear();
        self.accumulated_timed_tokens.clear();
    }

    pub fn get_timed_transcript(&self, mode: TimestampMode) -> TranscriptionResult {
        let text = self.get_transcript();
        let tokens = process_timestamps(&self.accumulated_timed_tokens, mode);
        TranscriptionResult { text, tokens }
    }

    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .copied()
            .filter(|&token| token < self.blank_id)
            .collect();
        self.vocab.decode(&valid)
    }

    pub fn transcribe_audio(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        self.transcribe_offline(audio, sample_rate, channels, None)
            .map(|result| result.text)
    }

    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let (audio, spec) = load_audio(audio_path)?;
        self.transcribe_audio(audio, spec.sample_rate, spec.channels)
    }

    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        self.audio_buffer.extend_from_slice(audio_chunk);
        self.process_ready_chunks(false)
    }

    pub fn flush(&mut self) -> Result<String> {
        self.process_ready_chunks(true)
    }

    fn process_ready_chunks(&mut self, flush: bool) -> Result<String> {
        let mut emitted = String::new();
        let chunk_samples = self.streaming_config.chunk_samples();
        let right_context_samples = self.streaming_config.right_context_samples();

        loop {
            let total_received = self.buffer_start_sample + self.audio_buffer.len();
            let ready = if flush {
                total_received > self.next_chunk_start_sample
            } else {
                total_received
                    >= self.next_chunk_start_sample + chunk_samples + right_context_samples
            };

            if !ready {
                break;
            }

            let (window_audio, left_encoder_frames, chunk_encoder_frames) =
                self.build_window_audio(self.next_chunk_start_sample, total_received, flush);
            if chunk_encoder_frames == 0 {
                break;
            }

            let features = self.extract_streaming_features(window_audio)?;
            let (encoded, encoded_len) = self.model.run_encoder(&features)?;

            let available_frames = (encoded_len as usize).min(encoded.shape()[2]);
            let start_frame = left_encoder_frames.min(available_frames);
            let end_frame = (start_frame + chunk_encoder_frames).min(available_frames);

            let absolute_frame_offset =
                self.next_chunk_start_sample / (HOP_LENGTH * SUBSAMPLING_FACTOR);
            let tokens =
                self.decode_encoder_frames(&encoded, start_frame, end_frame, absolute_frame_offset)?;
            self.accumulated_tokens
                .extend(tokens.iter().map(|(id, _)| *id));
            self.accumulated_timed_tokens
                .extend(self.tokens_to_timed(&tokens));
            emitted.push_str(&self.decode_incremental_tokens(&tokens));

            self.next_chunk_start_sample += chunk_samples;
            self.trim_audio_buffer();

            if flush && total_received <= self.next_chunk_start_sample {
                break;
            }
        }

        Ok(emitted)
    }

    fn build_window_audio(
        &self,
        chunk_start_sample: usize,
        total_received: usize,
        flush: bool,
    ) -> (Vec<f32>, usize, usize) {
        let left_context_samples = self.streaming_config.left_context_samples();
        let chunk_samples = self.streaming_config.chunk_samples();
        let right_context_samples = self.streaming_config.right_context_samples();

        let available_left = chunk_start_sample.saturating_sub(self.buffer_start_sample);
        let available_left = available_left.min(left_context_samples);
        let available_main = total_received.saturating_sub(chunk_start_sample).min(chunk_samples);
        let available_right = if flush {
            total_received
                .saturating_sub(chunk_start_sample + available_main)
                .min(right_context_samples)
        } else {
            right_context_samples
        };

        let window_start = chunk_start_sample.saturating_sub(available_left);
        let window_end = chunk_start_sample + available_main + available_right;
        let total_window_samples = window_end.saturating_sub(window_start);

        let left_encoder_frames = (available_left / HOP_LENGTH) / SUBSAMPLING_FACTOR;
        let chunk_encoder_frames = (available_main / HOP_LENGTH) / SUBSAMPLING_FACTOR;

        let mut window = vec![0.0f32; total_window_samples];
        let buffer_end = self.buffer_start_sample + self.audio_buffer.len();
        let copy_start = window_start.max(self.buffer_start_sample);
        let copy_end = window_end.min(buffer_end);

        if copy_end > copy_start {
            let src_start = copy_start - self.buffer_start_sample;
            let dst_start = copy_start - window_start;
            let len = copy_end - copy_start;
            window[dst_start..dst_start + len]
                .copy_from_slice(&self.audio_buffer[src_start..src_start + len]);
        }

        (window, left_encoder_frames, chunk_encoder_frames)
    }

    fn extract_streaming_features(&self, window_audio: Vec<f32>) -> Result<Array2<f32>> {
        self.extract_features(window_audio, SAMPLE_RATE as u32, 1)
    }

    fn trim_audio_buffer(&mut self) {
        let keep_from = self
            .next_chunk_start_sample
            .saturating_sub(self.streaming_config.left_context_samples());
        if keep_from <= self.buffer_start_sample {
            return;
        }

        let drop = keep_from - self.buffer_start_sample;
        if drop == 0 {
            return;
        }

        if drop >= self.audio_buffer.len() {
            self.audio_buffer.clear();
            self.buffer_start_sample = keep_from;
            return;
        }

        self.audio_buffer.drain(0..drop);
        self.buffer_start_sample = keep_from;
    }

    fn decode_encoder_frames(
        &mut self,
        encoder_out: &Array3<f32>,
        start_frame: usize,
        end_frame: usize,
        absolute_frame_offset: usize,
    ) -> Result<Vec<(usize, usize)>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];
        let end_frame = end_frame.min(encoder_out.shape()[2]);

        for frame_idx in start_frame..end_frame {
            let frame = encoder_out
                .slice(ndarray::s![0, .., frame_idx])
                .to_owned()
                .to_shape((1, hidden_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape encoder frame: {e}")))?
                .to_owned();

            let absolute_frame = absolute_frame_offset + (frame_idx - start_frame);

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let (token_id, new_state_1, new_state_2) = self.model.run_decoder(
                    &frame,
                    self.last_token,
                    &self.state_1,
                    &self.state_2,
                )?;

                if token_id == self.blank_id {
                    break;
                }

                tokens.push((token_id, absolute_frame));
                self.last_token = token_id as i32;
                self.state_1 = new_state_1;
                self.state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    fn encoder_frame_to_seconds(frame: usize) -> f32 {
        (frame * SUBSAMPLING_FACTOR * HOP_LENGTH) as f32 / SAMPLE_RATE as f32
    }

    fn tokens_to_timed(&self, tokens: &[(usize, usize)]) -> Vec<TimedToken> {
        tokens
            .iter()
            .filter(|(id, _)| *id < self.blank_id)
            .map(|&(id, frame)| TimedToken {
                text: self.vocab.decode_single(id),
                start: Self::encoder_frame_to_seconds(frame),
                end: Self::encoder_frame_to_seconds(frame + 1),
            })
            .collect()
    }

    fn decode_incremental_tokens(&self, tokens: &[(usize, usize)]) -> String {
        let mut text = String::new();
        for &(token, _) in tokens {
            if token < self.blank_id {
                text.push_str(&self.vocab.decode_single(token));
            }
        }
        text
    }

    fn transcribe_offline(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.reset();

        let features = self.extract_features(audio, sample_rate, channels)?;
        let (encoded, encoded_len) = self.model.run_encoder(&features)?;
        let frame_count = (encoded_len as usize).min(encoded.shape()[2]);
        let tokens = self.decode_encoder_frames(&encoded, 0, frame_count, 0)?;
        self.accumulated_tokens = tokens.iter().map(|(id, _)| *id).collect();
        self.accumulated_timed_tokens = self.tokens_to_timed(&tokens);

        let text = self.get_transcript();
        let timed = match mode {
            Some(m) => process_timestamps(&self.accumulated_timed_tokens, m),
            None => self.accumulated_timed_tokens.clone(),
        };

        Ok(TranscriptionResult {
            text,
            tokens: timed,
        })
    }

    fn extract_features(
        &self,
        mut audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Array2<f32>> {
        if sample_rate != self.preprocessor_config.sampling_rate as u32 {
            return Err(Error::Audio(format!(
                "Audio sample rate {} doesn't match expected {}. Please resample your audio first.",
                sample_rate, self.preprocessor_config.sampling_rate
            )));
        }

        if channels > 1 {
            audio = audio
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect();
        }

        let emphasized = Self::apply_preemphasis(&audio);
        let spectrogram = self.stft(&emphasized)?;
        let mel = self.mel_filterbank.dot(&spectrogram);
        let mel_log = mel.mapv(|value| (value + 2.0f64.powi(-24)).ln());
        let mut features = mel_log.t().mapv(|value| value as f32);

        let num_frames = features.shape()[0];
        let num_features = features.shape()[1];
        if num_frames <= 1 {
            return Ok(features);
        }

        for feature_idx in 0..num_features {
            let mean = features
                .column(feature_idx)
                .iter()
                .map(|&value| value as f64)
                .sum::<f64>()
                / num_frames as f64;

            let variance = features
                .column(feature_idx)
                .iter()
                .map(|&value| {
                    let delta = value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (num_frames as f64 - 1.0);

            let std = variance.sqrt() as f32 + 1e-5;
            let mut column = features.column_mut(feature_idx);
            for value in &mut column {
                *value = (*value - mean as f32) / std;
            }
        }

        Ok(features)
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f64> {
        if audio.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(audio.len());
        result.push(audio[0] as f64);
        for index in 1..audio.len() {
            result.push(audio[index] as f64 - PREEMPHASIS as f64 * audio[index - 1] as f64);
        }
        result
    }

    fn stft(&self, audio: &[f64]) -> Result<Array2<f64>> {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(N_FFT);

        let pad_amount = N_FFT / 2;
        let mut padded = vec![0.0f64; pad_amount];
        padded.extend_from_slice(audio);
        padded.resize(padded.len() + pad_amount, 0.0);

        let num_frames = (padded.len().saturating_sub(N_FFT)) / HOP_LENGTH + 1;
        let freq_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f64>::zeros((freq_bins, num_frames));

        let mut input = vec![0.0f64; N_FFT];
        let mut output = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;

            input.fill(0.0);
            let frame = &padded[start..start + WIN_LENGTH];
            for (idx, &value) in frame.iter().enumerate() {
                input[idx] = value * self.window[idx];
            }

            r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
                .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

            for (bin_idx, value) in output.iter().enumerate().take(freq_bins) {
                spectrogram[[bin_idx, frame_idx]] = value.norm_sqr();
            }
        }

        Ok(spectrogram)
    }

    fn create_window() -> Vec<f64> {
        (0..WIN_LENGTH)
            .map(|index| 0.5 - 0.5 * ((2.0 * PI * index as f64) / (WIN_LENGTH - 1) as f64).cos())
            .collect()
    }

    fn create_mel_filterbank() -> Array2<f64> {
        let freq_bins = N_FFT / 2 + 1;
        let mut filterbank = Array2::<f64>::zeros((FEATURE_SIZE, freq_bins));
        let fmax = SAMPLE_RATE as f64 / 2.0;
        let mel_min = Self::hz_to_mel_slaney(0.0);
        let mel_max = Self::hz_to_mel_slaney(fmax);

        let mel_points: Vec<f64> = (0..=FEATURE_SIZE + 1)
            .map(|index| {
                Self::mel_to_hz_slaney(
                    mel_min + (mel_max - mel_min) * index as f64 / (FEATURE_SIZE + 1) as f64,
                )
            })
            .collect();

        let fft_freqs: Vec<f64> = (0..freq_bins)
            .map(|index| index as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
            .collect();
        let fdiff: Vec<f64> = mel_points.windows(2).map(|window| window[1] - window[0]).collect();

        for mel_idx in 0..FEATURE_SIZE {
            for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
                let lower = (freq - mel_points[mel_idx]) / fdiff[mel_idx];
                let upper = (mel_points[mel_idx + 2] - freq) / fdiff[mel_idx + 1];
                filterbank[[mel_idx, freq_idx]] = 0.0f64.max(lower.min(upper));
            }

            let enorm = 2.0 / (mel_points[mel_idx + 2] - mel_points[mel_idx]);
            for freq_idx in 0..freq_bins {
                filterbank[[mel_idx, freq_idx]] *= enorm;
            }
        }

        filterbank
    }

    fn hz_to_mel_slaney(hz: f64) -> f64 {
        const F_SP: f64 = 200.0 / 3.0;
        const MIN_LOG_HZ: f64 = 1000.0;
        const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
        const LOG_STEP: f64 = 0.06875177742094912;

        if hz < MIN_LOG_HZ {
            hz / F_SP
        } else {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
        }
    }

    fn mel_to_hz_slaney(mel: f64) -> f64 {
        const F_SP: f64 = 200.0 / 3.0;
        const MIN_LOG_HZ: f64 = 1000.0;
        const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
        const LOG_STEP: f64 = 0.06875177742094912;

        if mel < MIN_LOG_MEL {
            mel * F_SP
        } else {
            MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
        }
    }
}

impl Transcriber for ParakeetUnified {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.transcribe_offline(audio, sample_rate, channels, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::UnifiedStreamingConfig;

    #[test]
    fn default_streaming_profile_aligns_to_subsampling() {
        let config = UnifiedStreamingConfig::default().validate().unwrap();
        assert_eq!(config.left_context_frames(), 560);
        assert_eq!(config.chunk_frames(), 56);
        assert_eq!(config.right_context_frames(), 56);
        assert_eq!(config.left_context_encoder_frames(), 70);
        assert_eq!(config.chunk_encoder_frames(), 7);
    }
}
