//! Cohere Transcribe ASR engine.
//!
//! 2B parameter Conformer encoder + lightweight Transformer decoder.
//! Takes raw 16 kHz mono f32 audio, returns transcribed text.
//! Supports 14 languages via explicit language selection.
//!
//! Consumes a HuggingFace-standard ONNX export: the encoder takes
//! pre-computed log-mel features and the decoder is a merged graph using
//! the standard `past_key_values.N.{decoder,encoder}.{key,value}` cache
//! convention.
//!
//! [`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
//! is one such export (FP32, FP16, INT8, and INT4 variants available).
//! To produce your own from the upstream PyTorch checkpoint, install
//! [`optimum`](https://github.com/huggingface/optimum) and run:
//!
//! ```sh
//! optimum-cli export onnx \
//!     --model CohereLabs/cohere-transcribe-03-2026 \
//!     --task automatic-speech-recognition-with-past \
//!     ./cohere-onnx
//! ```
//!
//! No custom export script is needed - the `cohere_asr` model type is
//! supported by Optimum's standard exporter.

use crate::audio::extract_features_raw;
use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_cohere::{CohereEncoderOutput, CohereModel, CoherePastKv, N_MELS};
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// Special token literals that drive the decoder prompt.
const TOKEN_STARTOFTRANSCRIPT: &str = "<|startoftranscript|>";
const TOKEN_ENDOFTEXT: &str = "<|endoftext|>";
const TOKEN_PNC: &str = "<|pnc|>";
const TOKEN_NOPNC: &str = "<|nopnc|>";
const TOKEN_NOTIMESTAMP: &str = "<|notimestamp|>";
const TOKEN_ITN: &str = "<|itn|>";
const TOKEN_NOITN: &str = "<|noitn|>";

/// Hard upper bound on output tokens enforced by the model
/// (`max_position_embeddings = 1024`). The user-configurable
/// `max_decode_tokens` cannot exceed this.
const MAX_DECODE_TOKENS_LIMIT: usize = 1024;

/// Default maximum output tokens per transcription. 512 is enough for
/// ~40 seconds of typical speech at the model's tokenisation rate, which
/// covers the training range (`max_audio_clip_s = 35`).
const DEFAULT_MAX_DECODE_TOKENS: usize = 512;

/// Maximum audio duration the model was trained on, as recorded in
/// `preprocessor_config.json` (`max_audio_clip_s`). Audio longer than this
/// will still run but transcription quality degrades beyond the training
/// range. Exposed via [`CohereASR::max_audio_duration_secs`] for callers
/// that want to chunk long audio.
const MAX_AUDIO_DURATION_SECS: f32 = 35.0;

// The 14 languages officially supported by Cohere Transcribe
// (cohere-transcribe-03-2026). The tokenizer contains `<|xx|>` placeholders
// for ~180 ISO codes but only these have trained weights.
// See https://docs.cohere.com/docs/transcribe.
const SUPPORTED_LANGUAGES: &[&str] = &[
    "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
];

/// Cohere Transcribe ASR engine.
pub struct CohereASR {
    model: CohereModel,
    tokenizer: Tokenizer,
    /// Mel/STFT parameters loaded from `preprocessor_config.json`.
    preprocessor: PreprocessorConfig,
    /// Map of supported ISO 639-1 language code -> language token id.
    lang_tokens: HashMap<String, i64>,
    /// Cached prompt token ids: startoftranscript / endoftext / pnc / nopnc /
    /// notimestamp / itn / noitn.
    sot_id: i64,
    eos_id: i64,
    pnc_id: i64,
    nopnc_id: i64,
    notimestamp_id: i64,
    itn_id: i64,
    noitn_id: i64,
    /// Maximum number of tokens to generate per `transcribe_audio` call.
    /// Defaults to [`DEFAULT_MAX_DECODE_TOKENS`] (512). Capped at
    /// [`MAX_DECODE_TOKENS_LIMIT`] (1024).
    max_decode_tokens: usize,
}

impl CohereASR {
    /// Load the Cohere Transcribe model from a directory.
    ///
    /// The directory must contain (flat or under `onnx/`):
    /// - one of `encoder_model[_quantized|_fp16].onnx` (+ `.onnx_data` companions)
    /// - one of `decoder_model_merged[_quantized|_fp16].onnx` (+ `.onnx_data`)
    /// - `tokenizer.json`
    /// - `preprocessor_config.json`
    ///
    /// This layout matches the [`onnx-community/cohere-transcribe-03-2026-ONNX`](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
    /// HF repository.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let model = CohereModel::from_pretrained(model_dir, exec)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Config(format!(
                "Missing tokenizer.json in {}",
                model_dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer.json: {e}")))?;

        let preprocessor_path = model_dir.join("preprocessor_config.json");
        if !preprocessor_path.exists() {
            return Err(Error::Config(format!(
                "Missing preprocessor_config.json in {}",
                model_dir.display()
            )));
        }
        let preprocessor: PreprocessorConfig =
            serde_json::from_str(&std::fs::read_to_string(&preprocessor_path).map_err(|e| {
                Error::Config(format!("Failed to read preprocessor_config.json: {e}"))
            })?)
            .map_err(|e| Error::Config(format!("Failed to parse preprocessor_config.json: {e}")))?;

        if preprocessor.feature_size != N_MELS {
            return Err(Error::Config(format!(
                "Cohere Transcribe expects feature_size=128, got {}",
                preprocessor.feature_size
            )));
        }

        let sot_id = require_token(&tokenizer, TOKEN_STARTOFTRANSCRIPT)?;
        let eos_id = require_token(&tokenizer, TOKEN_ENDOFTEXT)?;
        let pnc_id = require_token(&tokenizer, TOKEN_PNC)?;
        let nopnc_id = require_token(&tokenizer, TOKEN_NOPNC)?;
        let notimestamp_id = require_token(&tokenizer, TOKEN_NOTIMESTAMP)?;
        let itn_id = require_token(&tokenizer, TOKEN_ITN)?;
        let noitn_id = require_token(&tokenizer, TOKEN_NOITN)?;

        let mut lang_tokens = HashMap::with_capacity(SUPPORTED_LANGUAGES.len());
        for code in SUPPORTED_LANGUAGES {
            let lit = format!("<|{code}|>");
            if let Some(id) = tokenizer.token_to_id(&lit) {
                lang_tokens.insert((*code).to_string(), id as i64);
            }
        }
        if lang_tokens.is_empty() {
            return Err(Error::Tokenizer(
                "No supported language tokens found in tokenizer.json".into(),
            ));
        }

        Ok(Self {
            model,
            tokenizer,
            preprocessor,
            lang_tokens,
            sot_id,
            eos_id,
            pnc_id,
            nopnc_id,
            notimestamp_id,
            itn_id,
            noitn_id,
            max_decode_tokens: DEFAULT_MAX_DECODE_TOKENS,
        })
    }

    /// Maximum audio duration (in seconds) the model was trained on.
    /// Audio longer than this will still be transcribed but quality
    /// degrades outside the training range. Callers that process long
    /// recordings should chunk into segments of this length or shorter.
    pub fn max_audio_duration_secs(&self) -> f32 {
        MAX_AUDIO_DURATION_SECS
    }

    /// Current maximum number of tokens the decoder will emit per call.
    pub fn max_decode_tokens(&self) -> usize {
        self.max_decode_tokens
    }

    /// Set the maximum number of tokens the decoder will emit per call.
    /// Values above the model's hard limit (1024) are clamped.
    pub fn set_max_decode_tokens(&mut self, max: usize) {
        self.max_decode_tokens = max.clamp(1, MAX_DECODE_TOKENS_LIMIT);
    }

    /// Transcribe raw 16 kHz mono f32 audio samples.
    ///
    /// `language` is an ISO 639-1 code (e.g. `"en"`, `"fr"`, `"de"`, `"ja"`).
    /// `punctuation` controls whether output includes punctuation and
    /// capitalisation. `itn` enables inverse text normalisation
    /// (e.g. "twenty three" -> "23").
    pub fn transcribe_audio(
        &mut self,
        audio: &[f32],
        language: &str,
        punctuation: bool,
        itn: bool,
    ) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

        let lang_token = self.lang_tokens.get(language).copied().ok_or_else(|| {
            Error::Config(format!(
                "Unsupported language '{}'. Supported: {:?}",
                language,
                self.supported_languages()
            ))
        })?;

        // 1. Mel features. extract_features_raw returns [T, N_MELS] after
        //    preemphasis + STFT + log-mel + per-feature normalisation, which
        //    matches the CohereAsrFeatureExtractor pipeline. We add a batch
        //    axis to get [1, T, N_MELS] for the encoder.
        //
        // `as_standard_layout().to_owned()` is required because `insert_axis`
        // on a view may produce non-standard strides, but ort::TensorRef
        // needs C-contiguous memory.
        let mel_2d = extract_features_raw(
            audio.to_vec(),
            self.preprocessor.sampling_rate as u32,
            1,
            &self.preprocessor,
        )?;
        let mel_3d = mel_2d.insert_axis(Axis(0)).as_standard_layout().to_owned();

        // 2. Encoder
        let encoder_out = self.model.run_encoder(&mel_3d)?;

        // 3. Build decoder prompt:
        //    [<|startoftranscript|>, <|lang|>, <|pnc|>/<|nopnc|>,
        //     <|notimestamp|>, <|itn|>/<|noitn|>]
        let pnc_token = if punctuation {
            self.pnc_id
        } else {
            self.nopnc_id
        };
        let itn_token = if itn { self.itn_id } else { self.noitn_id };
        let prompt = vec![
            self.sot_id,
            lang_token,
            pnc_token,
            self.notimestamp_id,
            itn_token,
        ];

        // 4. Greedy decode loop
        let token_ids = self.decode_greedy(&prompt, &encoder_out)?;

        // 5. Detokenise (skip special tokens)
        let text = self
            .tokenizer
            .decode(
                &token_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| Error::Tokenizer(format!("Failed to decode tokens: {e}")))?;

        // Strip leading stray punctuation the decoder sometimes emits
        // before the first real token.
        let cleaned = text
            .trim()
            .trim_start_matches(['.', '?', '!', ','])
            .trim()
            .to_string();

        Ok(cleaned)
    }

    /// Greedy autoregressive decode using the merged decoder's growing
    /// `past_key_values` cache. The first call feeds the prompt and lets
    /// the model populate the cross-attention encoder cache; subsequent
    /// calls feed one token at a time.
    fn decode_greedy(
        &mut self,
        prompt: &[i64],
        encoder_out: &CohereEncoderOutput,
    ) -> Result<Vec<i64>> {
        let mut past_kv = CoherePastKv::empty();
        let mut output_tokens: Vec<i64> = Vec::new();

        // First step: feed entire prompt
        let prompt_tensor = Array2::from_shape_vec((1, prompt.len()), prompt.to_vec())
            .map_err(|e| Error::Model(format!("Prompt tensor shape error: {e}")))?;
        let (logits, new_past) =
            self.model
                .run_decoder_step(&prompt_tensor, &past_kv, encoder_out)?;
        past_kv = new_past;

        let mut next_token = argmax(logits.as_slice().unwrap());
        if next_token == self.eos_id {
            return Ok(output_tokens);
        }
        output_tokens.push(next_token);

        // Continue one token at a time up to the configured max.
        for _ in 1..self.max_decode_tokens {
            let token_tensor = Array2::from_shape_vec((1, 1), vec![next_token])
                .map_err(|e| Error::Model(format!("Token tensor shape error: {e}")))?;
            let (logits, new_past) =
                self.model
                    .run_decoder_step(&token_tensor, &past_kv, encoder_out)?;
            past_kv = new_past;

            next_token = argmax(logits.as_slice().unwrap());
            if next_token == self.eos_id {
                break;
            }
            output_tokens.push(next_token);

            // Detect n-gram repetition: if the last N tokens match a
            // previous sequence the model is stuck in a loop.
            if let Some(repeat_len) = find_ngram_repetition(&output_tokens, 8) {
                output_tokens.truncate(output_tokens.len() - repeat_len);
                break;
            }
        }

        Ok(output_tokens)
    }

    /// Sorted list of supported ISO 639-1 language codes.
    pub fn supported_languages(&self) -> Vec<String> {
        let mut langs: Vec<String> = self.lang_tokens.keys().cloned().collect();
        langs.sort();
        langs
    }
}

/// Look up a special token id by literal, returning a clear error if it's
/// not present in the tokenizer vocabulary.
fn require_token(tokenizer: &Tokenizer, literal: &str) -> Result<i64> {
    tokenizer
        .token_to_id(literal)
        .map(|id| id as i64)
        .ok_or_else(|| Error::Tokenizer(format!("Tokenizer is missing required token {literal}")))
}

/// Check if the token sequence ends with a repeated n-gram of length
/// `>= min_len`. Returns `Some(repeat_len)` if the last `repeat_len` tokens
/// are an exact copy of the preceding segment.
fn find_ngram_repetition(tokens: &[i64], min_len: usize) -> Option<usize> {
    let n = tokens.len();
    if n < min_len * 2 {
        return None;
    }
    for repeat_len in min_len..=(n / 2) {
        let tail = &tokens[n - repeat_len..];
        let prev = &tokens[n - 2 * repeat_len..n - repeat_len];
        if tail == prev {
            return Some(repeat_len);
        }
    }
    None
}

/// Greedy argmax over a slice of f32 logits.
fn argmax(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn test_supported_languages_count() {
        // Cohere Transcribe officially ships trained weights for 14 languages
        assert_eq!(SUPPORTED_LANGUAGES.len(), 14);
    }

    #[test]
    fn test_ngram_repetition_detection() {
        // No repetition
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 5, 6, 7, 8], 4), None);

        // Repeated 4-gram: [1,2,3,4] appears twice
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 1, 2, 3, 4], 4), Some(4));

        // Repeated 8-gram
        let mut tokens = vec![10, 20, 30, 40, 50, 60, 70, 80];
        tokens.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(find_ngram_repetition(&tokens, 8), Some(8));

        // Too short to detect
        assert_eq!(find_ngram_repetition(&[1, 2, 1, 2], 4), None);
    }
}
