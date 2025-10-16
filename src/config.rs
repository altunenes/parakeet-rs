use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub feature_extractor_type: String,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub padding_side: String,
    pub padding_value: f32,
    pub preemphasis: f32,
    pub processor_class: String,
    pub return_attention_mask: bool,
    pub sampling_rate: usize,
    pub win_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

impl PreprocessorConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: PreprocessorConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}
