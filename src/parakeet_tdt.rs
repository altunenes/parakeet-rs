use hound::WavSpec;

use crate::audio;
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::decoder_tdt::ParakeetTDTDecoder;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_tdt::ParakeetTDTModel;
use crate::vocab::Vocabulary;
use std::path::{Path, PathBuf};

/// Parakeet TDT model for multilingual ASR
pub struct ParakeetTDT {
    model: ParakeetTDTModel,
    decoder: ParakeetTDTDecoder,
    preprocessor_config: PreprocessorConfig,
    model_dir: PathBuf,
}

impl ParakeetTDT {
    /// Load Parakeet TDT model from path with optional configuration.
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder-model.onnx, decoder_joint-model.onnx, and vocab.txt
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(Error::Config(format!(
                "TDT model path must be a directory: {}",
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

        // TDT-specific preprocessor config (128 features instead of 80)
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

        let model = ParakeetTDTModel::from_pretrained(path, exec_config)?;
        let vocab = Vocabulary::from_file(&vocab_path)?;
        let decoder = ParakeetTDTDecoder::from_vocab(vocab);

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            model_dir: path.to_path_buf(),
        })
    }

    /// Transcribes audio samples into a transcription result with timestamps.
    ///
    /// # Arguments
    ///
    /// * `audio`: A vector of 32-bit floating-point values representing the audio signal.
    /// * `spec`: WavSpec struct containing information about the waveform (e.g., channels, sample rate).
    ///
    /// # Returns
    ///
    /// This function returns a `TranscriptionResult` which includes the transcribed text along with durations for timestamping.
    pub fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        spec: WavSpec,
    ) -> Result<TranscriptionResult> {
        let features = audio::extract_features(audio, spec, &self.preprocessor_config)?;
        let (tokens, frame_indices, durations) = self.model.forward(features)?;

        self.decoder.decode_with_timestamps(
            &tokens,
            &frame_indices,
            &durations,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
        )
    }

    /// Transcribe an audio file with token-level timestamps
    ///
    /// # Arguments
    ///
    /// * `audio_path` - A path to the audio file that needs to be transcribed.
    ///
    /// # Returns
    ///
    /// This function returns a `TranscriptionResult` which includes the transcribed text along with durations for timestamping.
    pub fn transcribe_file<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
    ) -> Result<TranscriptionResult> {
        let audio_path = audio_path.as_ref();
        let (audio, spec) = audio::load_audio(audio_path)?;

        self.transcribe_samples(audio, spec)
    }

    /// Transcribes multiple audio files in batch.
    ///
    /// # Arguments
    ///
    /// * `audio_paths`: A slice of paths to the audio files that need to be transcribed.
    ///
    /// # Returns
    ///
    /// This function returns a `TranscriptionResult` which includes the transcribed text along with durations for timestamping.
    pub fn transcribe_file_batch<P: AsRef<Path>>(
        &mut self,
        audio_paths: &[P],
    ) -> Result<Vec<TranscriptionResult>> {
        let mut results = Vec::with_capacity(audio_paths.len());
        for path in audio_paths {
            let result = self.transcribe_file(path)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Get model directory path
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}
