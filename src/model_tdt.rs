use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use std::path::{Path, PathBuf};

/// TDT model configs
#[derive(Debug, Clone)]
pub struct TDTModelConfig {
    pub vocab_size: usize,
}

impl TDTModelConfig {
    /// Create config with specified vocab size
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

pub struct ParakeetTDTModel {
    encoder: Session,
    decoder_joint: Session,
    config: TDTModelConfig,
}

impl ParakeetTDTModel {
    /// Load TDT model from directory containing encoder and decoder_joint ONNX files
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing encoder and decoder_joint ONNX files
    /// * `exec_config` - Execution configuration for ONNX runtime
    /// * `vocab_size` - Vocabulary size (number of tokens including blank)
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Find encoder and decoder_joint files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir)?;

        let config = TDTModelConfig::new(vocab_size);

        // Load encoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder_joint
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_joint_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
        })
    }
    //file names simply from: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main
    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        // fallback
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("encoder") && name.ends_with(".onnx") {
                        return Ok(path);
                    }
                }
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }


    fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint-model.onnx",
            "decoder_joint-model.int8.onnx",
            "decoder_joint.onnx",
            "decoder-model.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No decoder_joint model found in {}",
            dir.display()
        )))
    }

    /// Run greedy decoding - returns (token_ids, frame_indices, durations)
    pub fn forward(
        &mut self,
        features: Array2<f32>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        // Run encoder
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;

        // Run greedy decoding with decoder_joint
        let (tokens, frame_indices, durations) = self.greedy_decode(&encoder_out, encoder_len)?;

        Ok((tokens, frame_indices, durations))
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let batch_size = 1;
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        // TDT encoder expects (batch, features, time) not (batch, time, features)
        let input = features
            .t()
            .to_shape((batch_size, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to reshape encoder input: {e}")))?
            .to_owned();

        let input_length = Array1::from_vec(vec![time_steps as i64]);

        let input_value = ort::value::Value::from_array(input)?;
        let length_value = ort::value::Value::from_array(input_length)?;

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => input_value,
            "length" => length_value
        ))?;

        let encoder_out = &outputs["outputs"];
        let encoder_lens = &outputs["encoded_lengths"];

        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let (_, lens_data) = encoder_lens
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder lengths: {e}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {shape_dims:?}"
            )));
        }

        let b = shape_dims[0] as usize;
        let t = shape_dims[1] as usize;
        let d = shape_dims[2] as usize;

        let encoder_array = Array3::from_shape_vec((b, t, d), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create encoder array: {e}")))?;

        // TDT encoder outputs [batch, encoder_dim, time] directly
        Ok((encoder_array, lens_data[0]))
    }

    fn greedy_decode(
        &mut self,
        encoder_out: &Array3<f32>,
        _encoder_len: i64,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        // encoder_out shape: [batch, encoder_dim, time]
        let encoder_dim = encoder_out.shape()[1];
        let time_steps = encoder_out.shape()[2];
        let vocab_size = self.config.vocab_size;
        let max_tokens_per_step = 10;
        let blank_id = vocab_size - 1;

        // States: (num_layers=2, batch=1, hidden_dim=640)
        let mut state_h = Array3::<f32>::zeros((2, 1, 640));
        let mut state_c = Array3::<f32>::zeros((2, 1, 640));

        let mut tokens = Vec::new();
        let mut frame_indices = Vec::new();
        let mut durations = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;
        let mut last_emitted_token = blank_id as i32;

        // Frame-by-frame RNN-T/TDT greedy decoding
        while t < time_steps {
            // Get single encoder frame: slice [0, :, t] and reshape to [1, encoder_dim, 1]
            let frame = encoder_out.slice(ndarray::s![0, .., t]).to_owned();
            let frame_reshaped = frame
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            // Current token for prediction network
            let targets = Array2::from_shape_vec((1, 1), vec![last_emitted_token])
                .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

            // Run decoder_joint
            let outputs = self.decoder_joint.run(ort::inputs!(
                "encoder_outputs" => ort::value::Value::from_array(frame_reshaped)?,
                "targets" => ort::value::Value::from_array(targets)?,
                "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))?,
                "input_states_1" => ort::value::Value::from_array(state_h.clone())?,
                "input_states_2" => ort::value::Value::from_array(state_c.clone())?
            ))?;

            // Extract logits
            let (_, logits_data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

            // TDT outputs vocab_size + 5 durations
            let vocab_logits: Vec<f32> = logits_data.iter().take(vocab_size).copied().collect();
            let duration_logits: Vec<f32> = logits_data.iter().skip(vocab_size).copied().collect();

            let token_id = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);

            let duration_step = if !duration_logits.is_empty() {
                duration_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                0
            };

            // Check if blank token
            if token_id != blank_id {
                // Update states when we emit a token
                if let Ok((h_shape, h_data)) =
                    outputs["output_states_1"].try_extract_tensor::<f32>()
                {
                    let dims = h_shape.as_ref();
                    state_h = Array3::from_shape_vec(
                        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                        h_data.to_vec(),
                    )
                    .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;
                }
                if let Ok((c_shape, c_data)) =
                    outputs["output_states_2"].try_extract_tensor::<f32>()
                {
                    let dims = c_shape.as_ref();
                    state_c = Array3::from_shape_vec(
                        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                        c_data.to_vec(),
                    )
                    .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;
                }

                tokens.push(token_id);
                frame_indices.push(t);
                durations.push(duration_step);
                last_emitted_token = token_id as i32;
                emitted_tokens += 1;
            }
            // When duration > 0, skip frames according to duration prediction
            // Otherwise advance by 1 on blank or when max tokens reached
            if duration_step > 0 {
                t += duration_step;
                emitted_tokens = 0;
            } else if token_id == blank_id || emitted_tokens >= max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok((tokens, frame_indices, durations))
    }
}
