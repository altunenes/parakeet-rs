use crate::decoder::TranscriptionResult;
use crate::error::Result;
use crate::vocab::Vocabulary;

/// TDT greedy decoder for Parakeet TDT models
#[derive(Debug)]
pub struct ParakeetTDTDecoder {
    vocab: Vocabulary,
}

impl ParakeetTDTDecoder {
    /// Load decoder from vocab file
    pub fn from_vocab(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    /// Decode tokens with timestamps
    /// For TDT models, greedy decoding is done in the model, here we just convert to text
    pub fn decode_with_timestamps(
        &self,
        tokens: &[usize],
        frame_indices: &[usize],
        _durations: &[usize],
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        let mut result_tokens = Vec::new();
        let mut full_text = String::new();
        // TDT encoder does 8x subsampling
        let encoder_stride = 8;

        for (i, &token_id) in tokens.iter().enumerate() {
            if let Some(token_text) = self.vocab.id_to_text(token_id) {
                let frame = frame_indices[i];
                let start = (frame * encoder_stride * hop_length) as f32 / sample_rate as f32;
                let end = if i + 1 < frame_indices.len() {
                    (frame_indices[i + 1] * encoder_stride * hop_length) as f32 / sample_rate as f32
                } else {
                    start + 0.01
                };

                // Handle SentencePiece format (▁ prefix for word start)
                let display_text = token_text.replace('▁', " ");

                // Skip special tokens
                if !(token_text.starts_with('<')
                    && token_text.ends_with('>')
                    && token_text != "<unk>")
                {
                    full_text.push_str(&display_text);

                    result_tokens.push(crate::decoder::TimedToken {
                        text: display_text,
                        start,
                        end,
                    });
                }
            }
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            tokens: result_tokens,
        })
    }
}
