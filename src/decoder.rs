use crate::error::{Error, Result};
use ndarray::Array2;
use std::path::Path;

// Token with its timestamp information
// start and end are in seconds
#[derive(Debug, Clone)]
pub struct TimedToken {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<TimedToken>,
}

// CTC decoder for parakeet-ctc-0.6b model with token-level timestamps
pub struct ParakeetDecoder {
    tokenizer: tokenizers::Tokenizer,
    pad_token_id: usize,
}

impl ParakeetDecoder {
    pub fn from_pretrained<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref();

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {e}")))?;

        // Hardcoded pad_token_id for Parakeet-CTC-0.6b (constant across all models: please see def configs jsons: https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main)
        let pad_token_id = 1024;

        Ok(Self {
            tokenizer,
            pad_token_id,
        })
    }

    pub fn decode(&self, logits: &Array2<f32>) -> Result<String> {
        let time_steps = logits.shape()[0];

        let mut token_ids = Vec::new();
        for t in 0..time_steps {
            let logits_t = logits.row(t);
            let max_idx = logits_t
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            token_ids.push(max_idx as u32);
        }

        let collapsed = self.ctc_collapse(&token_ids);

        let text = self
            .tokenizer
            .decode(&collapsed, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?;

        Ok(text)
    }

    fn ctc_collapse(&self, token_ids: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut prev_token: Option<u32> = None;

        for &token_id in token_ids {
            if token_id == self.pad_token_id as u32 {
                prev_token = Some(token_id);
                continue;
            }

            if Some(token_id) != prev_token {
                result.push(token_id);
            }

            prev_token = Some(token_id);
        }

        result
    }

    // CTC collapse with frame tracking for timestamps
    fn ctc_collapse_with_frames(&self, token_ids: &[(u32, usize)]) -> Vec<(u32, usize, usize)> {
        let mut result: Vec<(u32, usize, usize)> = Vec::new();
        let mut prev_token: Option<u32> = None;

        for &(token_id, frame) in token_ids.iter() {
            if token_id == self.pad_token_id as u32 {
                prev_token = Some(token_id);
                continue;
            }

            if Some(token_id) != prev_token {
                if let Some(prev) = prev_token {
                    if prev != self.pad_token_id as u32 {
                        // End previous token
                        if let Some(last) = result.last_mut() {
                            last.2 = frame;
                        }
                    }
                }
                // Start new token
                result.push((token_id, frame, frame));
            }

            prev_token = Some(token_id);
        }

        // Close last token
        if let Some(last) = result.last_mut() {
            last.2 = token_ids.len();
        }

        result
    }

    // Decode with token-level timestamps
    // hop_length and sample_rate are needed to convert frames to seconds
    pub fn decode_with_timestamps(
        &self,
        logits: &Array2<f32>,
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        let time_steps = logits.shape()[0];

        let mut token_ids_with_frames = Vec::new();
        for t in 0..time_steps {
            let logits_t = logits.row(t);
            let max_idx = logits_t
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            token_ids_with_frames.push((max_idx as u32, t));
        }

        // CTC collapse with frame tracking
        let collapsed_with_frames = self.ctc_collapse_with_frames(&token_ids_with_frames);

        // Extract just token IDs for decoding
        let token_ids: Vec<u32> = collapsed_with_frames.iter().map(|(id, _, _)| *id).collect();

        // Decode full text
        let full_text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {e}")))?;

        // Progressive decode to detect word boundaries
        // BPE tokenizers only add spaces when decoding sequences, not individual tokens
        let mut timed_tokens = Vec::new();
        let mut prev_decode = String::new();

        for (i, (_token_id, start_frame, end_frame)) in collapsed_with_frames.iter().enumerate() {
            // Decode from start up to and including current token
            let token_ids_so_far: Vec<u32> = collapsed_with_frames[0..=i]
                .iter()
                .map(|(id, _, _)| *id)
                .collect();

            if let Ok(curr_decode) = self.tokenizer.decode(&token_ids_so_far, true) {
                // Find what this token added
                let added_text = if curr_decode.len() > prev_decode.len() {
                    &curr_decode[prev_decode.len()..]
                } else {
                    ""
                };

                if !added_text.is_empty() {
                    let start_time = (*start_frame * hop_length) as f32 / sample_rate as f32;
                    let end_time = (*end_frame * hop_length) as f32 / sample_rate as f32;

                    timed_tokens.push(TimedToken {
                        text: added_text.to_string(),
                        start: start_time,
                        end: end_time,
                    });
                }

                prev_decode = curr_decode;
            }
        }

        Ok(TranscriptionResult {
            text: full_text,
            tokens: timed_tokens,
        })
    }

    // Stub - falls back to greedy decoding. Full beam search with language model is TODO.
    pub fn decode_with_beam_search(
        &self,
        logits: &Array2<f32>,
        _beam_width: usize,
    ) -> Result<String> {
        self.decode(logits)
    }

    pub fn pad_token_id(&self) -> usize {
        self.pad_token_id
    }
}
