use serde::{Deserialize, Serialize};

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

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 80,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["ParakeetForCTC".to_string()],
            vocab_size: 1025,
            pad_token_id: 1024,
        }
    }
}
