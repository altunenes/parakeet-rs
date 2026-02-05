use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_nemotron::{NemotronEncoderCache, NemotronModel, NemotronModelConfig};
use ndarray::{s, Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::fs::File;
use std::io::Read;
use std::path::Path;

// Nemotron 0.6B model constants
// note that those numbers are coming from offical impl. and of course my onnx export decisions.
// Buffer logic and cache slicing strategy derived from:
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/utils/streaming_utils.py
// https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/audio_preprocessing.py
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

// Encoder arch
const NUM_ENCODER_LAYERS: usize = 24;
const HIDDEN_DIM: usize = 1024;
const LEFT_CONTEXT: usize = 70;
const CONV_CONTEXT: usize = 8;

// Decoder
const VOCAB_SIZE: usize = 1024;
const BLANK_ID: usize = 1024;
const DECODER_LSTM_DIM: usize = 640;

// Streaming chunk config
const CHUNK_SIZE: usize = 56;
const PRE_ENCODE_CACHE: usize = 9;

/// Minimal SentencePiece vocabulary loader.
/// Parses the protobuf .model file to extract token strings.
/// Note that, our vocab.rs cannot parse protobuf format. I haven't test it with digit spacing yet, at least for this initial impl.
pub struct SentencePieceVocab {
    pieces: Vec<String>,
}

impl SentencePieceVocab {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to open tokenizer.model: {e}")))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Tokenizer(format!("Failed to read tokenizer.model: {e}")))?;

        let pieces = Self::parse_sentencepiece_model(&data)?;
        Ok(Self { pieces })
    }

    fn parse_sentencepiece_model(data: &[u8]) -> Result<Vec<String>> {
        let mut pieces = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize > data.len() {
                        break;
                    }

                    let piece_data = &data[pos..pos + len as usize];
                    pos += len as usize;

                    if let Ok(piece) = Self::parse_piece_message(piece_data) {
                        pieces.push(piece);
                    }
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        if pieces.is_empty() {
            return Err(Error::Tokenizer("No tokens found in model".into()));
        }

        Ok(pieces)
    }

    fn parse_piece_message(data: &[u8]) -> Result<String> {
        let mut pos = 0;
        let mut piece = String::new();

        while pos < data.len() {
            let (field_header, bytes_read) = Self::read_varint(&data[pos..])?;
            pos += bytes_read;

            let field_num = field_header >> 3;
            let wire_type = field_header & 0x7;

            match (field_num, wire_type) {
                (1, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;

                    if pos + len as usize <= data.len() {
                        piece = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
                    }
                    pos += len as usize;
                }
                (_, 0) => {
                    let (_, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, bytes_read) = Self::read_varint(&data[pos..])?;
                    pos += bytes_read + len as usize;
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        }

        Ok(piece)
    }

    fn read_varint(data: &[u8]) -> Result<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut pos = 0;

        while pos < data.len() && pos < 10 {
            let byte = data[pos];
            result |= ((byte & 0x7F) as u64) << shift;
            pos += 1;

            if byte & 0x80 == 0 {
                return Ok((result, pos));
            }
            shift += 7;
        }

        Err(Error::Tokenizer("Invalid varint".into()))
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        for &id in ids {
            if id < self.pieces.len() {
                let piece = &self.pieces[id];
                let decoded = piece.replace('\u{2581}', " ");
                result.push_str(&decoded);
            }
        }
        result.trim_start().to_string()
    }

    pub fn decode_single(&self, id: usize) -> String {
        if id < self.pieces.len() {
            self.pieces[id].replace('\u{2581}', " ")
        } else {
            String::new()
        }
    }

    pub fn size(&self) -> usize {
        self.pieces.len()
    }
}

/// Nemotron streaming ASR model (0.6B parameters).
/// We dont apply mel normalization unlike others...
pub struct Nemotron {
    model: NemotronModel,
    vocab: SentencePieceVocab,
    encoder_cache: NemotronEncoderCache,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    mel_basis: Array2<f32>,
    window: Vec<f32>,
    /// Raw audio sample buffer for proper mel computation
    audio_buffer: Vec<f32>,
    /// How many audio samples have been processed (converted to mel and sent to encoder)
    audio_processed: usize,
    chunk_idx: usize,
    accumulated_tokens: Vec<usize>,
}

impl Nemotron {
    /// Load Nemotron model from directory.
    ///
    /// Required files:
    /// - encoder.onnx + encoder.onnx.data
    /// - decoder_joint.onnx
    /// - tokenizer.model
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;

        let model_config = NemotronModelConfig {
            num_encoder_layers: NUM_ENCODER_LAYERS,
            hidden_dim: HIDDEN_DIM,
            left_context: LEFT_CONTEXT,
            conv_context: CONV_CONTEXT,
            decoder_lstm_dim: DECODER_LSTM_DIM,
            decoder_lstm_layers: 2,
            vocab_size: VOCAB_SIZE,
            blank_id: BLANK_ID,
        };

        let exec = exec_config.unwrap_or_default();
        let model = NemotronModel::from_pretrained(path, exec, model_config)?;

        let encoder_cache =
            NemotronEncoderCache::with_dims(NUM_ENCODER_LAYERS, LEFT_CONTEXT, HIDDEN_DIM, CONV_CONTEXT);

        Ok(Self {
            model,
            vocab,
            encoder_cache,
            state_1: Array3::zeros((2, 1, DECODER_LSTM_DIM)),
            state_2: Array3::zeros((2, 1, DECODER_LSTM_DIM)),
            last_token: BLANK_ID as i32,
            mel_basis: crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE),
            window: Self::create_window(),
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            accumulated_tokens: Vec::new(),
        })
    }

    /// Reset all state for new utterance
    pub fn reset(&mut self) {
        self.encoder_cache =
            NemotronEncoderCache::with_dims(NUM_ENCODER_LAYERS, LEFT_CONTEXT, HIDDEN_DIM, CONV_CONTEXT);
        self.state_1.fill(0.0);
        self.state_2.fill(0.0);
        self.last_token = BLANK_ID as i32;
        self.audio_buffer.clear();
        self.audio_processed = 0;
        self.chunk_idx = 0;
        self.accumulated_tokens.clear();
    }

    /// Get the full accumulated transcript
    pub fn get_transcript(&self) -> String {
        let valid: Vec<usize> = self
            .accumulated_tokens
            .iter()
            .filter(|&&t| t < VOCAB_SIZE)
            .copied()
            .collect();
        self.vocab.decode(&valid)
    }

    /// note that, offline transcription for testing/debugging and for some curious ppl :-). with following function too (transcribe_audio)
    pub fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<String> {
        let (audio, spec) = crate::audio::load_audio(audio_path)?;

        let audio = if spec.channels > 1 {
            audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            audio
        };

        self.transcribe_audio(&audio)
    }

    /// Transcribe audio samples (non-streaming)
    pub fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        self.reset();

        let mel = self.compute_mel_spectrogram(audio);
        let total_frames = mel.shape()[1];

        if total_frames == 0 {
            return Ok(String::new());
        }

        let mut all_tokens: Vec<usize> = Vec::new();
        let mut buffer_idx = 0;
        let mut chunk_idx = 0;

        while buffer_idx < total_frames {
            let chunk_end = (buffer_idx + CHUNK_SIZE).min(total_frames);
            let main_len = chunk_end - buffer_idx;

            let expected_size = PRE_ENCODE_CACHE + CHUNK_SIZE;
            let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

            // Fill pre-encode cache from previous frames
            if chunk_idx > 0 && buffer_idx >= PRE_ENCODE_CACHE {
                let cache_start = buffer_idx - PRE_ENCODE_CACHE;
                for f in 0..PRE_ENCODE_CACHE {
                    for m in 0..N_MELS {
                        chunk_data[m * expected_size + f] = mel[[m, cache_start + f]];
                    }
                }
            }

            // Fill main chunk
            for f in 0..main_len {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = mel[[m, buffer_idx + f]];
                }
            }

            let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
                .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))?;

            let chunk_length = PRE_ENCODE_CACHE + main_len;

            let (encoded, enc_len, new_cache) =
                self.model
                    .run_encoder(&mel_chunk, chunk_length as i64, &self.encoder_cache)?;
            self.encoder_cache = new_cache;

            let new_tokens = self.decode_chunk(&encoded, enc_len as usize)?;
            all_tokens.extend(new_tokens);

            buffer_idx += CHUNK_SIZE;
            chunk_idx += 1;
        }

        let valid_tokens: Vec<usize> = all_tokens.into_iter().filter(|&t| t < VOCAB_SIZE).collect();

        Ok(self.vocab.decode(&valid_tokens))
    }

    /// Stream transcribe a chunk of audio (call repeatedly for real-time).
    ///
    /// This buffers raw audio and computes mel spectrograms over the full buffer
    /// to avoid edge effects at chunk boundaries.
    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        // Append raw audio to buffer
        self.audio_buffer.extend_from_slice(audio_chunk);

        // Calculate how many mel frames we can produce from buffered audio
        // mel_frames = 1 + (audio_len + 2*pad - win_length) / hop_length
        // For center=true padding, we need at least win_length samples to get 1 frame
        let total_audio = self.audio_buffer.len();
        if total_audio < WIN_LENGTH {
            return Ok(String::new());
        }

        // Compute mel spectrogram over the ENTIRE audio buffer
        let full_mel = self.compute_mel_spectrogram(&self.audio_buffer);
        let total_mel_frames = full_mel.shape()[1];

        // Calculate how many mel frames correspond to processed audio
        // Each CHUNK_SIZE mel frames = CHUNK_SIZE * HOP_LENGTH audio samples
        let processed_mel_frames = self.audio_processed / HOP_LENGTH;

        // Check if we have enough NEW frames to process a chunk
        let available_new_frames = total_mel_frames.saturating_sub(processed_mel_frames);
        if available_new_frames < CHUNK_SIZE {
            return Ok(String::new());
        }

        // Build encoder input chunk
        let expected_size = PRE_ENCODE_CACHE + CHUNK_SIZE;
        let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

        // Determine the mel frame range for this chunk
        let is_first_chunk = self.chunk_idx == 0;
        let main_start = processed_mel_frames;
        let _main_end = main_start + CHUNK_SIZE;

        if is_first_chunk {
            // First chunk: zero-pad for pre-encode cache
            for f in 0..CHUNK_SIZE.min(total_mel_frames) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
            // Subsequent chunks: include pre-encode cache from previous frames
            let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
            let cache_frames = main_start - cache_start;
            let cache_offset = PRE_ENCODE_CACHE - cache_frames;

            // Fill pre-encode cache
            for f in 0..cache_frames {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + cache_offset + f] = full_mel[[m, cache_start + f]];
                }
            }

            // Fill main chunk
            for f in 0..CHUNK_SIZE.min(total_mel_frames - main_start) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = full_mel[[m, main_start + f]];
                }
            }
        }

        let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
            .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))?;

        let (encoded, enc_len, new_cache) =
            self.model
                .run_encoder(&mel_chunk, expected_size as i64, &self.encoder_cache)?;
        self.encoder_cache = new_cache;

        let tokens = self.decode_chunk(&encoded, enc_len as usize)?;
        self.accumulated_tokens.extend(&tokens);

        // Advance processed position
        self.audio_processed += CHUNK_SIZE * HOP_LENGTH;
        self.chunk_idx += 1;

        // Trim audio buffer to keep memory bounded
        // Keep enough for pre-encode cache context
        let keep_samples = (PRE_ENCODE_CACHE + CHUNK_SIZE) * HOP_LENGTH + WIN_LENGTH;
        if self.audio_buffer.len() > keep_samples * 2 {
            let remove = self.audio_buffer.len() - keep_samples;
            // Adjust processed counter since we're removing from the start
            let actual_remove = remove.min(self.audio_processed);
            self.audio_buffer.drain(0..actual_remove);
            self.audio_processed -= actual_remove;
        }

        let mut result = String::new();
        for &t in &tokens {
            if t < VOCAB_SIZE {
                result.push_str(&self.vocab.decode_single(t));
            }
        }
        Ok(result)
    }

    fn decode_chunk(&mut self, encoder_out: &Array3<f32>, enc_frames: usize) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];
        let max_symbols_per_step = 10;

        for t in 0..enc_frames {
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame = frame
                .to_shape((1, hidden_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            for _ in 0..max_symbols_per_step {
                let (logits, new_state_1, new_state_2) =
                    self.model
                        .run_decoder(&frame, self.last_token, &self.state_1, &self.state_2)?;

                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }

                if max_idx == BLANK_ID {
                    break;
                }

                tokens.push(max_idx);
                self.last_token = max_idx as i32;
                self.state_1 = new_state_1;
                self.state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    /// Compute log mel spectrogram WITHOUT normalization. 
    /// I use capitals because this gave me some trouble on the Python side :(). I realized they dont use it later. 
    /// so offc nemo feeding raw log-mel spectrogram values (in decibels) directly to the encoder.
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Array2<f32> {
        if audio.is_empty() {
            return Array2::zeros((N_MELS, 0));
        }

        let preemph = Self::apply_preemphasis(audio);
        let spec = self.stft_center(&preemph);
        let mel = self.mel_basis.dot(&spec);

        mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln())
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(audio.len());
        result.push(audio[0]);

        for i in 1..audio.len() {
            let v = audio[i] - PREEMPH * audio[i - 1];
            result.push(if v.is_finite() { v } else { 0.0 });
        }

        result
    }

    fn stft_center(&self, audio: &[f32]) -> Array2<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let pad_amount = N_FFT / 2;
        let mut padded = vec![0.0f32; pad_amount];
        padded.extend_from_slice(audio);
        padded.extend(std::iter::repeat_n(0.0f32, pad_amount));

        let num_frames = if padded.len() >= WIN_LENGTH {
            1 + (padded.len() - WIN_LENGTH) / HOP_LENGTH
        } else {
            0
        };

        let freq_bins = N_FFT / 2 + 1;
        let mut spec = Array2::zeros((freq_bins, num_frames));

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            if start + WIN_LENGTH > padded.len() {
                break;
            }

            let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
            for i in 0..WIN_LENGTH {
                buffer[i] = Complex::new(padded[start + i] * self.window[i], 0.0);
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

}
