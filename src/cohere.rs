//! Cohere Transcribe ASR engine.
//!
//! 2B parameter conformer encoder + lightweight Transformer decoder.
//! Takes raw 16kHz mono f32 audio, returns transcribed text.
//! Supports 14 languages via explicit language selection.

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_cohere::{CohereEncoderOutput, CohereModel, CohereSelfCache};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

// Special token IDs (from tokens.txt)
const TOKEN_EOS: i64 = 3; // <|endoftext|>
const TOKEN_START: i64 = 4; // <|startoftranscript|>
const TOKEN_PNC: i64 = 5; // <|pnc|> (punctuation on)
const TOKEN_NOPNC: i64 = 6; // <|nopnc|> (punctuation off)
const TOKEN_ITN: i64 = 8; // <|itn|> (inverse text normalisation on)
const TOKEN_NOITN: i64 = 9; // <|noitn|> (inverse text normalisation off)
const TOKEN_NOTIMESTAMP: i64 = 11; // <|notimestamp|>

// First special token ID - anything below this in the output is a control token
const FIRST_SPECIAL_END: i64 = 256;

// Max output tokens before forced stop
const MAX_DECODE_TOKENS: usize = 1024;

/// Cohere Transcribe ASR engine.
pub struct CohereASR {
    model: CohereModel,
    vocab: Vec<String>,
    lang_tokens: HashMap<String, i64>,
}

impl CohereASR {
    /// Load the Cohere Transcribe model from a directory containing:
    /// - cohere-encoder.int8.onnx (+ .data file)
    /// - cohere-decoder.int8.onnx
    /// - tokens.txt
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = exec_config.unwrap_or_default();

        let model = CohereModel::from_pretrained(model_dir, config)?;

        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(Error::Config(format!(
                "Missing tokens.txt in {}",
                model_dir.display()
            )));
        }

        let (vocab, lang_tokens) = Self::load_tokens(&tokens_path)?;

        Ok(Self {
            model,
            vocab,
            lang_tokens,
        })
    }

    /// Transcribe raw 16kHz mono f32 audio samples.
    ///
    /// `language` is an ISO 639-1 code (e.g. "en", "fr", "de", "ja").
    /// `punctuation` controls whether output includes punctuation and capitalisation.
    /// `itn` enables inverse text normalisation (e.g. "twenty three" -> "23").
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

        let lang_token = self
            .lang_tokens
            .get(language)
            .copied()
            .ok_or_else(|| {
                Error::Config(format!(
                    "Unsupported language '{}'. Supported: {:?}",
                    language,
                    self.supported_languages()
                ))
            })?;

        // Run encoder
        let encoder_out = self.model.run_encoder(audio)?;

        // Build decoder prompt
        let pnc_token = if punctuation { TOKEN_PNC } else { TOKEN_NOPNC };
        let itn_token = if itn { TOKEN_ITN } else { TOKEN_NOITN };
        let prompt = vec![TOKEN_START, lang_token, pnc_token, TOKEN_NOTIMESTAMP, itn_token];

        // Autoregressive decode
        let token_ids = self.decode_greedy(&prompt, &encoder_out)?;

        // Convert tokens to text, filtering control tokens
        let text = self.tokens_to_text(&token_ids);

        Ok(text)
    }

    /// Greedy autoregressive decoding with KV cache.
    fn decode_greedy(
        &mut self,
        prompt: &[i64],
        encoder_out: &CohereEncoderOutput,
    ) -> Result<Vec<i64>> {
        let mut cache = CohereSelfCache::empty();
        let mut all_tokens: Vec<i64> = Vec::new();

        // First step: feed entire prompt
        let prompt_tensor =
            Array2::from_shape_vec((1, prompt.len()), prompt.to_vec())
                .map_err(|e| Error::Model(format!("Prompt tensor error: {e}")))?;

        let (logits, new_cache) =
            self.model
                .run_decoder_step(&prompt_tensor, &cache, encoder_out, 0)?;
        cache = new_cache;

        // Get first predicted token (last position in logits)
        let logits_row = logits.row(logits.nrows() - 1);
        let mut next_token = argmax(logits_row.as_slice().unwrap());

        if next_token == TOKEN_EOS {
            return Ok(all_tokens);
        }
        all_tokens.push(next_token);

        let mut offset = prompt.len() as i64;

        // Continue decoding one token at a time
        for _ in 1..MAX_DECODE_TOKENS {
            let token_tensor = Array2::from_shape_vec((1, 1), vec![next_token])
                .map_err(|e| Error::Model(format!("Token tensor error: {e}")))?;

            let (logits, new_cache) =
                self.model
                    .run_decoder_step(&token_tensor, &cache, encoder_out, offset)?;
            cache = new_cache;
            offset += 1;

            let logits_row = logits.row(logits.nrows() - 1);
            next_token = argmax(logits_row.as_slice().unwrap());

            if next_token == TOKEN_EOS {
                break;
            }
            all_tokens.push(next_token);

            // Detect n-gram repetition: if the last N tokens match a previous
            // sequence, the model is stuck in a loop. Check for repeated
            // sequences of length 8-32 tokens.
            if let Some(repeat_len) = find_ngram_repetition(&all_tokens, 8) {
                all_tokens.truncate(all_tokens.len() - repeat_len);
                break;
            }
        }

        Ok(all_tokens)
    }

    /// Convert token IDs to text, filtering out special/control tokens.
    fn tokens_to_text(&self, token_ids: &[i64]) -> String {
        let mut text = String::new();
        for &id in token_ids {
            // Skip control tokens (IDs 0-255 are special tokens and byte tokens)
            if id < FIRST_SPECIAL_END {
                continue;
            }
            if let Some(piece) = self.vocab.get(id as usize) {
                // Skip any remaining <|...|> special tokens
                if piece.starts_with("<|") && piece.ends_with("|>") {
                    continue;
                }
                text.push_str(piece);
            }
        }
        // SentencePiece: replace word boundary marker with space
        let result = text.replace('\u{2581}', " ");
        // Strip leading stray punctuation the model sometimes emits
        result.trim().trim_start_matches(['.', '?', '!', ',']).trim().to_string()
    }

    /// Load tokens.txt: "token_text token_id" per line.
    /// Also extracts language token mappings.
    fn load_tokens(path: &Path) -> Result<(Vec<String>, HashMap<String, i64>)> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Tokenizer(format!("Failed to read tokens.txt: {e}")))?;

        let mut vocab: Vec<String> = Vec::new();
        let mut lang_tokens: HashMap<String, i64> = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Format: "token_text id"
            let Some((token, id_str)) = line.rsplit_once(' ') else {
                continue;
            };
            let Ok(id) = id_str.parse::<usize>() else {
                continue;
            };

            // Extend vocab vector if needed
            while vocab.len() <= id {
                vocab.push(String::new());
            }
            vocab[id] = token.to_string();

            // Extract language tokens: <|xx|> where xx is a 2-letter ISO code
            if token.starts_with("<|") && token.ends_with("|>") && token.len() == 6 {
                let lang_code = &token[2..4];
                lang_tokens.insert(lang_code.to_string(), id as i64);
            }
        }

        if vocab.is_empty() {
            return Err(Error::Tokenizer("tokens.txt is empty".into()));
        }

        Ok((vocab, lang_tokens))
    }

    /// Return the list of supported language codes.
    pub fn supported_languages(&self) -> Vec<String> {
        let mut langs: Vec<String> = self.lang_tokens.keys().cloned().collect();
        langs.sort();
        langs
    }
}

/// Check if the token sequence ends with a repeated n-gram.
/// Returns `Some(repeat_len)` if the last `repeat_len` tokens (>= `min_len`)
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
        .unwrap_or(TOKEN_EOS)
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
    fn test_load_tokens_format() {
        let dir = std::env::temp_dir().join("cohere_test_tokens");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.txt");
        std::fs::write(
            &path,
            "<unk> 0\n<|en|> 62\n<|fr|> 70\n\u{2581}hello 512\nworld 513\n",
        )
        .unwrap();

        let (vocab, lang_tokens) = CohereASR::load_tokens(&path).unwrap();
        assert_eq!(vocab[62], "<|en|>");
        assert_eq!(vocab[512], "\u{2581}hello");
        assert_eq!(vocab[513], "world");
        assert_eq!(lang_tokens["en"], 62);
        assert_eq!(lang_tokens["fr"], 70);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_ngram_repetition_detection() {
        // No repetition
        assert_eq!(find_ngram_repetition(&[1, 2, 3, 4, 5, 6, 7, 8], 4), None);

        // Repeated 4-gram: [1,2,3,4] appears twice
        assert_eq!(
            find_ngram_repetition(&[1, 2, 3, 4, 1, 2, 3, 4], 4),
            Some(4)
        );

        // Repeated 8-gram
        let mut tokens = vec![10, 20, 30, 40, 50, 60, 70, 80];
        tokens.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(find_ngram_repetition(&tokens, 8), Some(8));

        // Too short to detect
        assert_eq!(find_ngram_repetition(&[1, 2, 1, 2], 4), None);
    }
}
