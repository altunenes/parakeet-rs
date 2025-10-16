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
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        Self::from_pretrained_with_config(model_dir, ExecutionConfig::default())
    }

    pub fn from_pretrained_with_config<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let required_files = [
            "model.onnx",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ];

        for file in &required_files {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                return Err(Error::Config(format!(
                    "Required file '{file}' not found in model directory"
                )));
            }
        }

        let preprocessor_config =
            PreprocessorConfig::from_file(model_dir.join("preprocessor_config.json"))?;

        let model = ParakeetModel::from_pretrained_with_config(model_dir, exec_config)?;
        let decoder = ParakeetDecoder::from_pretrained(model_dir)?;

        Ok(Self {
            model,
            decoder,
            preprocessor_config,
            model_dir: model_dir.to_path_buf(),
        })
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
