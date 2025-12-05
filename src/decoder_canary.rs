use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Canary vocabulary with special tokens
#[derive(Debug, Clone)]
pub struct CanaryVocabulary {
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, i64>,
    pub eos_token_id: i64,
}

impl CanaryVocabulary {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to open vocab file: {e}")))?;

        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        for line in reader.lines() {
            let line =
                line.map_err(|e| Error::Config(format!("Failed to read vocab file: {e}")))?;

            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let token = parts[0].replace("\u{2581}", " ");
                let id: i64 = parts[1]
                    .parse()
                    .map_err(|e| Error::Config(format!("Invalid token ID in vocab: {e}")))?;

                if id as usize >= id_to_token.len() {
                    id_to_token.resize(id as usize + 1, String::new());
                }
                id_to_token[id as usize] = token.clone();
                token_to_id.insert(parts[0].to_string(), id);
            }
        }

        let eos_token_id = *token_to_id
            .get("<|endoftext|>")
            .ok_or_else(|| Error::Config("Missing <|endoftext|> token in vocabulary".into()))?;

        Ok(Self {
            id_to_token,
            token_to_id,
            eos_token_id,
        })
    }

    /// Get token ID by token string
    pub fn token_to_id(&self, token: &str) -> Option<i64> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string by ID
    pub fn id_to_token(&self, id: i64) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Decode token IDs to text (skip special tokens)
    pub fn decode(&self, token_ids: &[i64]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                self.id_to_token(id).and_then(|token| {
                    if token.starts_with("<|") {
                        None
                    } else {
                        Some(token)
                    }
                })
            })
            .collect::<Vec<_>>()
            .join("")
            .trim()
            .to_string()
    }

    /// Create initial prompt tokens for Canary transcription/translation
    /// Format: <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|source_lang|> <|target_lang|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    pub fn create_transcribe_prompt(
        &self,
        source_lang: Option<&str>,
        target_lang: Option<&str>,
    ) -> Result<Vec<i64>> {
        let src_lang = source_lang.unwrap_or("en");
        let tgt_lang = target_lang.unwrap_or("en");
        let src_lang_token = format!("<|{src_lang}|>");
        let tgt_lang_token = format!("<|{tgt_lang}|>");

        let tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            &src_lang_token, // Source language (what the audio is in)
            &tgt_lang_token, // Target language (what the output should be)
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ];

        tokens
            .iter()
            .map(|token| {
                self.token_to_id(token)
                    .ok_or_else(|| Error::Config(format!("Missing token {token} in vocabulary")))
            })
            .collect()
    }
}

/// Canary decoder for autoregressive decoding
pub struct CanaryDecoder {
    vocab: CanaryVocabulary,
    max_sequence_length: usize,
}

impl CanaryDecoder {
    pub fn from_pretrained<P: AsRef<Path>>(
        vocab_path: P,
        max_sequence_length: Option<usize>,
    ) -> Result<Self> {
        let vocab = CanaryVocabulary::from_file(vocab_path)?;
        Ok(Self {
            vocab,
            max_sequence_length: max_sequence_length.unwrap_or(1024),
        })
    }

    pub fn vocab(&self) -> &CanaryVocabulary {
        &self.vocab
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    pub fn decode_with_timestamps(
        &self,
        token_ids: &[i64],
        positions: &[usize],
        hop_length: usize,
        sampling_rate: usize,
    ) -> Result<TranscriptionResult> {
        let text = self.vocab.decode(token_ids);

        // Build tokens list with positions
        let mut tokens = Vec::new();

        for (i, &token_id) in token_ids.iter().enumerate() {
            if let Some(token_text) = self.vocab.id_to_token(token_id) {
                if token_text.starts_with("<|") {
                    continue;
                }

                let position = if i < positions.len() {
                    positions[i]
                } else {
                    positions.last().copied().unwrap_or(0)
                };

                let next_position = if i + 1 < positions.len() {
                    positions[i + 1]
                } else {
                    position + 1
                };

                let start_time = (position * hop_length) as f32 / sampling_rate as f32;
                let end_time = (next_position * hop_length) as f32 / sampling_rate as f32;

                // Create a token for each piece
                tokens.push(TimedToken {
                    text: token_text.to_string(),
                    start: start_time,
                    end: end_time,
                });
            }
        }

        Ok(TranscriptionResult { text, tokens })
    }
}
