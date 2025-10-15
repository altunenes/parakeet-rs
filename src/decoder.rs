use crate::error::{Error, Result};
use ndarray::Array2;
use std::path::Path;

// CTC decoder for parakeet-ctc-0.6b model.
// Note: This model doesn't support timestamps
pub struct ParakeetDecoder {
    tokenizer: tokenizers::Tokenizer,
    pad_token_id: usize,
}

impl ParakeetDecoder {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("tokenizer_config.json");

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {e}")))?;

        let config_content = std::fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_content)?;

        let pad_token_id = config
            .get("added_tokens_decoder")
            .and_then(|tokens| tokens.as_object())
            .and_then(|tokens| {
                tokens.iter().find_map(|(id, token)| {
                    if token.get("content")?.as_str()? == "<pad>" {
                        id.parse::<usize>().ok()
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(1024);

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
