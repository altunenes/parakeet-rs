use crate::decoder::TranscriptionResult;
use crate::error::Result;
use crate::vocab::Vocabulary;
use ndarray::Array2;

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

    /// Decode logits with timestamps
    /// For TDT models, the joint network outputs are processed greedily
    pub fn decode_with_timestamps(
        &self,
        logits: &Array2<f32>,
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        // Greedy decode: take argmax at each timestep
        let mut tokens = Vec::new();
        let mut frames = Vec::new();

        for (frame_idx, frame_logits) in logits.rows().into_iter().enumerate() {
            let max_idx = frame_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Skip blank tokens
            if max_idx != self.vocab.blank_id {
                tokens.push(max_idx);
                frames.push(frame_idx);
            }
        }

        // Collapse consecutive duplicates with frame tracking
        let collapsed = self.collapse_with_frames(&tokens, &frames);

        // Convert to text and timestamps
        let mut result_tokens = Vec::new();
        let mut full_text = String::new();

        for (token_id, start_frame, end_frame) in collapsed {
            if let Some(token_text) = self.vocab.id_to_text(token_id) {
                // Skip special tokens
                if token_text.starts_with('<') && token_text.ends_with('>') {
                    continue;
                }

                let start = (start_frame * hop_length) as f32 / sample_rate as f32;
                let end = (end_frame * hop_length) as f32 / sample_rate as f32;

                // Handle SentencePiece format (▁ prefix for word start)
                let display_text = token_text.replace('▁', " ");
                full_text.push_str(&display_text);

                result_tokens.push(crate::decoder::TimedToken {
                    text: display_text,
                    start,
                    end,
                });
            }
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            tokens: result_tokens,
        })
    }

    fn collapse_with_frames(
        &self,
        tokens: &[usize],
        frames: &[usize],
    ) -> Vec<(usize, usize, usize)> {
        let mut result: Vec<(usize, usize, usize)> = Vec::new();
        let mut prev_token: Option<usize> = None;

        for (&token_id, &frame) in tokens.iter().zip(frames.iter()) {
            if Some(token_id) != prev_token {
                if prev_token.is_some() {
                    if let Some(last) = result.last_mut() {
                        last.2 = frame;
                    }
                }
                result.push((token_id, frame, frame));
            }
            prev_token = Some(token_id);
        }

        // Set end frame for last token
        if let Some(last) = result.last_mut() {
            if let Some(&last_frame) = frames.last() {
                last.2 = last_frame;
            }
        }

        result
    }
}
