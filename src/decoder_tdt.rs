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
                let mut display_text = token_text.replace('▁', " ");

                // Insert space before pure digit tokens following words.
                // SentencePiece digits lack the ▁ prefix, causing "at60" instead of "at 60".
                // Skip single uppercase letters (A4, B12) but allow lowercase 'a' (article).
                if !full_text.is_empty()
                    && !display_text.starts_with(' ')
                    && display_text.chars().all(|c| c.is_ascii_digit())
                {
                    let trailing_letters: usize = full_text
                        .chars()
                        .rev()
                        .take_while(|c| c.is_alphabetic())
                        .count();
                    let last_char = full_text.chars().last();
                    let is_article_a = trailing_letters == 1 && last_char == Some('a');
                    if trailing_letters > 1 || is_article_a {
                        display_text.insert(0, ' ');
                    }
                }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(tokens: &[&str]) -> Vocabulary {
        Vocabulary {
            id_to_token: tokens.iter().map(|s| s.to_string()).collect(),
            _blank_id: 0,
        }
    }

    #[test]
    fn test_digit_spacing_after_word() {
        // Simulates "like 100" tokenized as ["▁like", "1", "0", "0"]
        let vocab = make_vocab(&["▁like", "1", "0", "0"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "like 100");
    }

    #[test]
    fn test_digit_spacing_after_article_a() {
        // Simulates "a 24" tokenized as ["▁a", "2", "4"]
        let vocab = make_vocab(&["▁a", "2", "4"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2], &[0, 1, 2], &[1, 1, 1], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "a 24");
    }

    #[test]
    fn test_no_spacing_after_single_uppercase() {
        // Simulates "A4" tokenized as ["▁A", "4"] - should stay together
        let vocab = make_vocab(&["▁A", "4"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1], &[0, 1], &[1, 1], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "A4");
    }

    #[test]
    fn test_no_spacing_after_symbol() {
        // Simulates "$100" tokenized as ["$", "1", "0", "0"]
        let vocab = make_vocab(&["$", "1", "0", "0"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "$100");
    }

    #[test]
    fn test_spacing_in_phrase() {
        // Simulates "In 2021" tokenized as ["▁In", "2", "0", "2", "1"]
        let vocab = make_vocab(&["▁In", "2", "0", "2", "1"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], &[1, 1, 1, 1, 1], 160, 16000)
            .unwrap();
        assert_eq!(result.text, "In 2021");
    }

    #[test]
    fn test_tokens_have_correct_spacing() {
        // Verify individual token texts have spacing applied
        let vocab = make_vocab(&["▁like", "1", "0", "0"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], 160, 16000)
            .unwrap();

        // Check token texts - first digit should have space prepended
        assert_eq!(result.tokens[0].text, " like");
        assert_eq!(result.tokens[1].text, " 1"); // Space added by heuristic
        assert_eq!(result.tokens[2].text, "0");
        assert_eq!(result.tokens[3].text, "0");
    }

    #[test]
    fn test_full_flow_with_timestamp_processing() {
        use crate::timestamps::{process_timestamps, TimestampMode};

        let vocab = make_vocab(&["▁like", "1", "0", "0", "▁bucks"]);
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3, 4],
                &[1, 1, 1, 1, 1],
                160,
                16000,
            )
            .unwrap();

        // Test Words mode (what Undertone likely uses)
        let words = process_timestamps(&result.tokens, TimestampMode::Words);
        let text: String = words.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ");
        assert_eq!(text, "like 100 bucks");

        // Test Tokens mode
        let tokens_text: String = result.tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(tokens_text.trim(), "like 100 bucks");
    }
}
