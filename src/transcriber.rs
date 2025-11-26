use crate::audio::load_audio;
use crate::decoder::TranscriptionResult;
use crate::error::Result;
use crate::timestamps::TimestampMode;
use std::path::Path;

/// Trait for common transcription functionality
pub trait Transcriber {
    /// Transcribe audio samples.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples as f32 values
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `mode` - Optional timestamp output mode (Tokens, Words, or Sentences)
    ///
    /// # Returns
    ///
    /// A `TranscriptionResult` containing the transcribed text and timestamps at the requested level.
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult>;

    /// Transcribe an audio file with timestamps
    ///
    /// # Arguments
    ///
    /// * `audio_path` - A path to the audio file that needs to be transcribed.
    /// * `mode` - Optional timestamp output mode (Tokens, Words, or Sentences)
    ///
    /// # Returns
    ///
    /// This function returns a `TranscriptionResult` which includes the transcribed text along with timestamps at the requested level.
    fn transcribe_file<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let audio_path = audio_path.as_ref();
        let (audio, spec) = load_audio(audio_path)?;

        self.transcribe_samples(audio, spec.sample_rate, spec.channels, mode)
    }

    /// Transcribes multiple audio files in batch.
    ///
    /// # Arguments
    ///
    /// * `audio_paths`: A slice of paths to the audio files that need to be transcribed.
    /// * `mode` - Optional timestamp output mode (Tokens, Words, or Sentences)
    ///
    /// # Returns
    ///
    /// This function returns a `TranscriptionResult` which includes the transcribed text along with timestamps at the requested level.
    fn transcribe_file_batch<P: AsRef<Path>>(
        &mut self,
        audio_paths: &[P],
        mode: Option<TimestampMode>,
    ) -> Result<Vec<TranscriptionResult>> {
        let mut results = Vec::with_capacity(audio_paths.len());
        for path in audio_paths {
            let result = self.transcribe_file(path, mode)?;
            results.push(result);
        }
        Ok(results)
    }
}
