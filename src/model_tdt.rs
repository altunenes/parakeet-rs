#[cfg(feature = "burn")]
use crate::burn_backend::tdt::{TdtDecoderJoint, TdtEncoder};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
#[cfg(feature = "ort")]
use ndarray::Array1;
use ndarray::{Array2, Array3};
#[cfg(feature = "ort")]
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
    encoder: Encoder,
    decoder_joint: DecoderJoint,
    config: TDTModelConfig,
}

/// The encoder graph, on whichever backend its execution configuration selects.
enum Encoder {
    #[cfg(feature = "ort")]
    Ort(Session),
    #[cfg(feature = "burn")]
    Burn(Box<TdtEncoder>),
}

/// The decoder/joint graph, on whichever backend its execution configuration selects.
enum DecoderJoint {
    #[cfg(feature = "ort")]
    Ort(Session),
    #[cfg(feature = "burn")]
    Burn(Box<TdtDecoderJoint>),
}

impl Encoder {
    fn load(path: &Path, config: &ExecutionConfig) -> Result<Self> {
        #[cfg(feature = "burn")]
        if config.execution_provider.is_burn() {
            return Ok(Self::Burn(Box::new(TdtEncoder::load(
                path,
                config.execution_provider,
            )?)));
        }
        #[cfg(feature = "ort")]
        {
            Ok(Self::Ort(config.build_session(path)?))
        }
        #[cfg(not(feature = "ort"))]
        {
            Err(Error::Config(format!(
                "{:?} needs the `ort` feature",
                config.execution_provider
            )))
        }
    }
}

impl DecoderJoint {
    #[cfg_attr(not(feature = "burn"), allow(unused_variables))]
    fn load(path: &Path, config: &ExecutionConfig, vocab_size: usize) -> Result<Self> {
        #[cfg(feature = "burn")]
        if config.execution_provider.is_burn() {
            return Ok(Self::Burn(Box::new(TdtDecoderJoint::load(
                path,
                config.execution_provider,
                vocab_size,
            )?)));
        }
        #[cfg(feature = "ort")]
        {
            Ok(Self::Ort(config.build_session(path)?))
        }
        #[cfg(not(feature = "ort"))]
        {
            Err(Error::Config(format!(
                "{:?} needs the `ort` feature",
                config.execution_provider
            )))
        }
    }
}

impl ParakeetTDTModel {
    /// Load a TDT model, running the encoder and the decoder/joint under separate execution
    /// configurations.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing encoder and decoder_joint ONNX files
    /// * `exec_config` - Execution configuration for the encoder session
    /// * `joint_config` - Execution configuration for the decoder/joint session
    /// * `vocab_size` - Vocabulary size (number of tokens including blank)
    pub fn from_pretrained_with_configs<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        joint_config: ExecutionConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        if vocab_size == 0 {
            return Err(Error::Config("TDT vocabulary is empty".into()));
        }
        let encoder_int8 = !exec_config.execution_provider.is_burn();
        let joint_int8 = !joint_config.execution_provider.is_burn();
        let encoder_path = Self::find_encoder(model_dir, encoder_int8)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir, joint_int8)?;

        let config = TDTModelConfig::new(vocab_size);

        let encoder = Encoder::load(&encoder_path, &exec_config)?;
        let decoder_joint = DecoderJoint::load(&decoder_joint_path, &joint_config, vocab_size)?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
        })
    }
    //file names simply from: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main
    /// `int8`: whether int8 exports may be picked (burn runs fp32 exports only).
    fn find_encoder(dir: &Path, int8: bool) -> Result<PathBuf> {
        let candidates = [
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() && (int8 || !candidate.contains(".int8.")) {
                return Ok(path);
            }
        }
        // fallback
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str())
                    && name.starts_with("encoder")
                    && name.ends_with(".onnx")
                    && (int8 || !name.contains(".int8."))
                {
                    return Ok(path);
                }
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder_joint(dir: &Path, int8: bool) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint-model.onnx",
            "decoder_joint-model.int8.onnx",
            "decoder_joint.onnx",
            "decoder-model.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() && (int8 || !candidate.contains(".int8.")) {
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

        match &mut self.encoder {
            #[cfg(feature = "ort")]
            Encoder::Ort(session) => {
                let input_length = Array1::from_vec(vec![time_steps as i64]);

                let input_value = ort::value::Value::from_array(input)?;
                let length_value = ort::value::Value::from_array(input_length)?;

                let outputs = session.run(ort::inputs!(
                    "audio_signal" => input_value,
                    "length" => length_value
                ))?;

                // TDT encoder outputs [batch, encoder_dim, time] directly
                let encoder_array =
                    crate::tensor_utils::extract_3d_f32(&outputs["outputs"], "encoder output")?;
                let encoded_len = crate::tensor_utils::extract_scalar_i64(
                    &outputs["encoded_lengths"],
                    "encoder lengths",
                )?;

                Ok((encoder_array, encoded_len))
            }
            #[cfg(feature = "burn")]
            Encoder::Burn(encoder) => encoder.run(input),
        }
    }

    /// Run the decoder/joint for one encoder frame `[1, encoder_dim, 1]`: returns the token and
    /// the duration it predicts, and updates the LSTM state when it emits a token.
    fn joint_step(
        &mut self,
        frame: Array3<f32>,
        last_token: i32,
        state_h: &mut Array3<f32>,
        state_c: &mut Array3<f32>,
    ) -> Result<(usize, usize)> {
        let vocab_size = self.config.vocab_size;
        let blank_id = vocab_size - 1;
        match &mut self.decoder_joint {
            #[cfg(feature = "ort")]
            DecoderJoint::Ort(session) => {
                // Current token for prediction network
                let targets = Array2::from_shape_vec((1, 1), vec![last_token])
                    .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

                let outputs = session.run(ort::inputs!(
                    "encoder_outputs" => ort::value::Value::from_array(frame)?,
                    "targets" => ort::value::Value::from_array(targets)?,
                    "target_length" => ort::value::Value::from_array(Array1::from_vec(vec![1i32]))?,
                    "input_states_1" => ort::value::Value::from_array(state_h.clone())?,
                    "input_states_2" => ort::value::Value::from_array(state_c.clone())?
                ))?;

                // Extract logits
                let (_, logits_data) = outputs["outputs"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
                let (token_id, duration_step) = pick(logits_data, vocab_size);

                if token_id != blank_id {
                    // Update states when we emit a token
                    if let Ok((h_shape, h_data)) =
                        outputs["output_states_1"].try_extract_tensor::<f32>()
                    {
                        let dims = h_shape.as_ref();
                        *state_h = Array3::from_shape_vec(
                            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                            h_data.to_vec(),
                        )
                        .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;
                    }
                    if let Ok((c_shape, c_data)) =
                        outputs["output_states_2"].try_extract_tensor::<f32>()
                    {
                        let dims = c_shape.as_ref();
                        *state_c = Array3::from_shape_vec(
                            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
                            c_data.to_vec(),
                        )
                        .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;
                    }
                }
                Ok((token_id, duration_step))
            }
            #[cfg(feature = "burn")]
            DecoderJoint::Burn(decoder) => {
                let step = decoder.step(frame.view(), last_token, state_h, state_c)?;
                let (token_id, duration_step) = pick(&step.logits, vocab_size);
                if token_id != blank_id {
                    *state_h = step.state_h;
                    *state_c = step.state_c;
                }
                Ok((token_id, duration_step))
            }
        }
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

            // Run decoder_joint
            let (token_id, duration_step) = self.joint_step(
                frame_reshaped,
                last_emitted_token,
                &mut state_h,
                &mut state_c,
            )?;

            // Check if blank token
            if token_id != blank_id {
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

/// The token and the duration a TDT joint output predicts: it holds `vocab_size` token logits
/// followed by the duration logits.
fn pick(logits: &[f32], vocab_size: usize) -> (usize, usize) {
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    };
    let (vocab_logits, duration_logits) = logits.split_at(vocab_size.min(logits.len()));
    let token_id = argmax(vocab_logits).unwrap_or(vocab_size - 1);
    let duration_step = argmax(duration_logits).unwrap_or(0);
    (token_id, duration_step)
}

#[cfg(test)]
mod tests {
    use super::pick;

    #[test]
    fn pick_splits_token_and_duration_logits() {
        // vocabulary of 3, then 2 duration logits
        assert_eq!(pick(&[0.1, 0.9, 0.2, 0.3, 0.7], 3), (1, 1));
        // no duration logits: duration 0
        assert_eq!(pick(&[0.1, 0.2, 0.9], 3), (2, 0));
    }
}
