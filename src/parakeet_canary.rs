use crate::audio;
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::decoder_canary::CanaryDecoder;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_canary::CanaryModel;
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use ndarray::{s, Array1, Array2, Array4};
use std::path::{Path, PathBuf};

/// Canary 1B multilingual ASR/AST model supporting 25 European languages https://huggingface.co/nvidia/canary-1b-v2
/// 
/// Canary requires both source and target languages:
/// - Transcription (ASR): source="sv", target="sv" → Swedish audio → Swedish text
/// - Translation (AST): source="sv", target="en" → Swedish audio → English text
///
/// Supported languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it,
/// lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
pub struct ParakeetCanary {
    model: CanaryModel,
    decoder: CanaryDecoder,
    preprocessor_config: PreprocessorConfig,
    model_dir: PathBuf,
    source_language: String,
    target_language: String,
}

impl ParakeetCanary {
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(Error::Config(format!(
                "Canary model path must be a directory: {}",
                path.display()
            )));
        }

        let vocab_path = path.join("vocab.txt");
        if !vocab_path.exists() {
            return Err(Error::Config(format!(
                "vocab.txt not found in {}",
                path.display()
            )));
        }

        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        };

        let exec_config = config.unwrap_or_default();
        let model = CanaryModel::from_pretrained(path, exec_config)?;
        let decoder = CanaryDecoder::from_pretrained(&vocab_path, Some(1024))?;

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            model_dir: path.to_path_buf(),
            source_language: "en".to_string(), // Default: English input
            target_language: "en".to_string(), // Default: English output
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }

    /// Set the source language (language of the INPUT audio)
    ///
    /// Supported languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it,
    /// lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
    pub fn set_source_language(&mut self, lang: &str) {
        self.source_language = lang.to_string();
    }

    /// Get the current source language
    pub fn source_language(&self) -> &str {
        &self.source_language
    }

    /// Set the target language (language of the OUTPUT text)
    ///
    /// Supported languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it,
    /// lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
    pub fn set_target_language(&mut self, lang: &str) {
        self.target_language = lang.to_string();
    }

    /// Get the current target language
    pub fn target_language(&self) -> &str {
        &self.target_language
    }

    /// Convenience method to set languages for transcription (source = target)
    ///
    /// Example: `canary.set_language("sv")` for Swedish audio → Swedish text
    pub fn set_language(&mut self, lang: &str) {
        self.source_language = lang.to_string();
        self.target_language = lang.to_string();
    }

    fn transcribe_internal(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        // Calculate audio duration
        let samples_per_channel = audio.len() / channels as usize;
        let duration_seconds = samples_per_channel as f32 / sample_rate as f32;

        // Canary models natively handle inputs up to ~40 seconds. For longer audio,
        // the input is split into 30-40s chunks (minimizing padding on the final chunk)
        // and processed sequentially.
        //
        // Note: The official NVIDIA NeMo implementation processes chunks in parallel.
        // This implementation processes them sequentially, which produces the same
        // results but may be slower for very long audio files.
        //
        // For recordings longer than one hour, the official implementation uses
        // consecutive hour-long segments. This is not yet implemented here.
        const MAX_CHUNK_DURATION: f32 = 35.0; // Use 35s to

        if duration_seconds <= MAX_CHUNK_DURATION {
            // Audio is short enough, process directly
            return self.transcribe_chunk(audio, sample_rate, channels, mode, 0.0);
        }

        // Long audio: split into chunks and process each
        let chunk_samples = (MAX_CHUNK_DURATION * sample_rate as f32 * channels as f32) as usize;
        let mut all_tokens = Vec::new();
        let mut full_text = String::new();

        let mut offset = 0;
        while offset < audio.len() {
            let chunk_end = (offset + chunk_samples).min(audio.len());
            let chunk = audio[offset..chunk_end].to_vec();

            // Calculate time offset for this chunk
            let time_offset = offset as f32 / (sample_rate as f32 * channels as f32);

            // Transcribe chunk
            let mut chunk_result =
                self.transcribe_chunk(chunk, sample_rate, channels, mode, time_offset)?;

            // Adjust timestamps by chunk offset
            for token in &mut chunk_result.tokens {
                token.start += time_offset;
                token.end += time_offset;
            }

            // Accumulate results
            if !full_text.is_empty() && !chunk_result.text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&chunk_result.text);
            all_tokens.extend(chunk_result.tokens);

            offset = chunk_end;
        }

        Ok(TranscriptionResult {
            text: full_text,
            tokens: all_tokens,
        })
    }

    /// Transcribes a single audio chunk.
    ///
    /// From NVIDIA documentation:
    /// "Canary models natively handle inputs up to ~40 seconds. For longer audio,
    /// the input is split into 30-40s chunks (minimizing padding on the final chunk)
    /// and processed in parallel. For recordings longer than one hour, processing
    /// occurs in consecutive hour-long segments."
    ///
    /// This function processes a single chunk using the Canary encoder-decoder architecture:
    /// 1. Extract 128-dim log-mel features (25ms window, 10ms stride)
    /// 2. Encode features using FastConformer encoder
    /// 3. Generate tokens autoregressively using Transformer decoder
    /// 4. Decode tokens to text with approximate timestamps
    fn transcribe_chunk(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
        _time_offset: f32,
    ) -> Result<TranscriptionResult> {
        let features =
            audio::extract_features_raw(audio, sample_rate, channels, &self.preprocessor_config)?;

        // Reshape features from (time, features) to (batch, features, time)
        let batch_size = 1;
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        let features_3d = features
            .t()
            .to_shape((batch_size, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to reshape features: {e}")))?
            .to_owned();

        let features_len = Array1::from_vec(vec![time_steps as i64]);

        // Encode
        let (encoder_embeddings, encoder_mask) = self.model.encode(features_3d, features_len)?;

        // Get initial prompt tokens with configured source and target languages
        let prompt_tokens = self
            .decoder
            .vocab()
            .create_transcribe_prompt(Some(&self.source_language), Some(&self.target_language))?;
        let mut tokens = prompt_tokens.clone();

        // Autoregressive decoding
        let batch_size = encoder_embeddings.shape()[0];
        let decoder_mems_shape = self.get_decoder_mems_shape()?;
        let mut decoder_mems =
            Array4::<f32>::zeros((decoder_mems_shape[0], batch_size, 0, decoder_mems_shape[3]));

        let max_len = self.decoder.max_sequence_length();
        let eos_token_id = self.decoder.vocab().eos_token_id;

        // Track token positions for timestamp calculation
        // Use encoder output sequence length for alignment
        let encoder_seq_len = encoder_embeddings.shape()[1];
        let mut token_positions = Vec::new();

        // Calculate encoder downsampling: time_steps -> encoder_seq_len
        let downsample_factor = time_steps as f32 / encoder_seq_len as f32;

        while tokens.len() < max_len {
            // prep input
            let input_ids = if decoder_mems.shape()[2] == 0 {
                // First iteration: use all tokens
                Array2::from_shape_vec((1, tokens.len()), tokens.clone())
                    .map_err(|e| Error::Model(format!("Failed to create input_ids: {e}")))?
            } else {
                // sub iterations: use only last token
                Array2::from_shape_vec((1, 1), vec![*tokens.last().unwrap()])
                    .map_err(|e| Error::Model(format!("Failed to create input_ids: {e}")))?
            };

            // Decode
            let (logits, new_decoder_mems) = self.model.decode(
                input_ids,
                encoder_embeddings.clone(),
                encoder_mask.clone(),
                decoder_mems,
            )?;

            decoder_mems = new_decoder_mems;

            // Get next token
            let last_logits = logits.slice(s![0, -1, ..]);
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap();

            if next_token == eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        // Calculate positions by distributing tokens evenly across encoder sequence
        // Then scale back to original time dimension accounting for downsampling
        let num_output_tokens = tokens.len() - prompt_tokens.len();
        token_positions.clear();
        for i in 0..num_output_tokens {
            let encoder_pos = i as f32 * encoder_seq_len as f32 / num_output_tokens as f32;
            let original_pos = (encoder_pos * downsample_factor) as usize;
            token_positions.push(original_pos);
        }

        // Decode tokens with timestamps (skip prompt)
        let output_tokens = &tokens[prompt_tokens.len()..];
        let mut result = self.decoder.decode_with_timestamps(
            output_tokens,
            &token_positions,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
        )?;

        // Apply timestamp mode if requested
        if let Some(timestamp_mode) = mode {
            use crate::timestamps::process_timestamps;
            result.tokens = process_timestamps(&result.tokens, timestamp_mode);

            result.text = result
                .tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
        }

        Ok(result)
    }

    fn get_decoder_mems_shape(&self) -> Result<Vec<usize>> {
        // Canary decoder_mems shape: [num_layers, batch_size, seq_len, hidden_size]
        // Canary 1B v2: 10 layers, 1024 hidden size
        Ok(vec![10, 1, 0, 1024])
    }
}

impl Transcriber for ParakeetCanary {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        self.transcribe_internal(audio, sample_rate, channels, mode)
    }
}
