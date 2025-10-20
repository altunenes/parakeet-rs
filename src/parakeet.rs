use crate::audio;
use crate::config::PreprocessorConfig;
use crate::decoder::{ParakeetDecoder, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model::ParakeetModel;
use std::path::{Path, PathBuf};

pub struct Parakeet {
    model: ParakeetModel,
    decoder: ParakeetDecoder,
    preprocessor_config: PreprocessorConfig,
    model_dir: PathBuf,
}

impl Parakeet {
    /// Load Parakeet model from path with optional configuration.
    ///
    /// # Arguments
    /// * `path` - Directory containing model files, or path to specific model file
    /// * `config` - Optional execution configuration (defaults to CPU if None)
    ///
    /// # Examples
    /// ```no_run
    /// use parakeet_rs::Parakeet;
    ///
    /// // Load from directory with CPU (default)
    /// let parakeet = Parakeet::from_pretrained(".", None)?;
    ///
    /// // Or load from specific model file
    /// let parakeet = Parakeet::from_pretrained("model_q4.onnx", None)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// For GPU acceleration, enable the corresponding feature (cuda, tensorrt, webgpu, etc.)
    /// and pass an `ExecutionConfig` with the desired execution provider.
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Determine if path is a directory or file
        let (model_path, tokenizer_path, model_dir) = if path.is_dir() {
            // Directory mode: auto-detect model file
            let model_path = Self::find_model_file(path)?;
            let tokenizer_path = path.join("tokenizer.json");
            (model_path, tokenizer_path, path.to_path_buf())
        } else if path.is_file() {
            // File mode: path points directly to model file
            let model_dir = path
                .parent()
                .ok_or_else(|| Error::Config("Invalid model path".to_string()))?;
            let tokenizer_path = model_dir.join("tokenizer.json");
            (path.to_path_buf(), tokenizer_path, model_dir.to_path_buf())
        } else {
            return Err(Error::Config(format!(
                "Path does not exist: {}",
                path.display()
            )));
        };

        // Check tokenizer exists
        if !tokenizer_path.exists() {
            return Err(Error::Config(format!(
                "Required file 'tokenizer.json' not found in {}",
                model_dir.display()
            )));
        }

        let preprocessor_config = PreprocessorConfig::default();
        let exec_config = config.unwrap_or_default();

        let model = ParakeetModel::from_pretrained_with_config(&model_path, exec_config)?;
        let decoder = ParakeetDecoder::from_pretrained(&tokenizer_path)?;

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            model_dir,
        })
    }

    fn find_model_file(dir: &Path) -> Result<PathBuf> {
        // Priority order: model.onnx > model_fp16.onnx > model_int8.onnx > model_q4.onnx
        let candidates = [
            "model.onnx",
            "model_fp16.onnx",
            "model_int8.onnx",
            "model_q4.onnx",
        ];

        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        // If none of the standard names found, search for any .onnx file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                    return Ok(path);
                }
            }
        }

        Err(Error::Config(format!(
            "No model file (*.onnx) found in directory: {}",
            dir.display()
        )))
    }

    pub fn transcribe<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<TranscriptionResult> {
        let audio_path = audio_path.as_ref();
        let features = audio::extract_features(audio_path, &self.preprocessor_config)?;
        let logits = self.model.forward(features)?;

        let result = self.decoder.decode_with_timestamps(
            &logits,
            self.preprocessor_config.hop_length,
            self.preprocessor_config.sampling_rate,
        )?;

        Ok(result)
    }

    pub fn transcribe_batch<P: AsRef<Path>>(
        &mut self,
        audio_paths: &[P],
    ) -> Result<Vec<TranscriptionResult>> {
        let mut results = Vec::with_capacity(audio_paths.len());
        for path in audio_paths {
            let result = self.transcribe(path)?;
            results.push(result);
        }
        Ok(results)
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }
}
