use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::path::Path;

/// Encoder cache state for Nemotron streaming inference.
#[derive(Clone)]
pub struct NemotronEncoderCache {
    /// [24, 1, 70, 1024] - 24 layers, batch=1, 70 frame lookback, 1024 features
    pub cache_last_channel: Array4<f32>,
    /// [24, 1, 1024, 8] - 24 layers, batch=1, 1024 features, 8 conv context
    pub cache_last_time: Array4<f32>,
    /// [1] - current cache length
    pub cache_last_channel_len: Array1<i64>,
}

impl Default for NemotronEncoderCache {
    fn default() -> Self {
        Self::new()
    }
}

impl NemotronEncoderCache {
    pub fn new() -> Self {
        Self {
            cache_last_channel: Array4::zeros((24, 1, 70, 1024)),
            cache_last_time: Array4::zeros((24, 1, 1024, 8)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }

    pub fn with_dims(num_layers: usize, left_context: usize, hidden_dim: usize, conv_context: usize) -> Self {
        Self {
            cache_last_channel: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
            cache_last_time: Array4::zeros((num_layers, 1, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

/// Nemotron ONNX wrapper.
/// we handle encoder and decoder_joint sessions separately.
pub struct NemotronModel {
    encoder: Session,
    decoder_joint: Session,
    pub config: NemotronModelConfig,
}

/// cfg for Nemotron model dims.
#[derive(Debug, Clone)]
pub struct NemotronModelConfig {
    pub num_encoder_layers: usize,
    pub hidden_dim: usize,
    pub left_context: usize,
    pub conv_context: usize,
    pub decoder_lstm_dim: usize,
    pub decoder_lstm_layers: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
}

impl Default for NemotronModelConfig {
    fn default() -> Self {
        Self {
            num_encoder_layers: 24,
            hidden_dim: 1024,
            left_context: 70,
            conv_context: 8,
            decoder_lstm_dim: 640,
            decoder_lstm_layers: 2,
            vocab_size: 1024,
            blank_id: 1024,
        }
    }
}

impl NemotronModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        config: NemotronModelConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = model_dir.join("encoder.onnx");
        let decoder_path = model_dir.join("decoder_joint.onnx");

        if !encoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing encoder.onnx in {}",
                model_dir.display()
            )));
        }
        if !decoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing decoder_joint.onnx in {}",
                model_dir.display()
            )));
        }

        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
        })
    }

    /// Run encoder with cache-aware streaming.
    /// i: mel features [1, n_mels, time], cache state
    /// o: (encoded [1, hidden_dim, time], new_cache)
    pub fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &NemotronEncoderCache,
    ) -> Result<(Array3<f32>, i64, NemotronEncoderCache)> {
        let length_arr = Array1::from_vec(vec![length]);

        let outputs = self.encoder.run(ort::inputs![
            "processed_signal" => ort::value::Value::from_array(features.clone())?,
            "processed_signal_length" => ort::value::Value::from_array(length_arr)?,
            "cache_last_channel" => ort::value::Value::from_array(cache.cache_last_channel.clone())?,
            "cache_last_time" => ort::value::Value::from_array(cache.cache_last_time.clone())?,
            "cache_last_channel_len" => ort::value::Value::from_array(cache.cache_last_channel_len.clone())?
        ])?;

        // [1, hidden_dim, time]
        let (shape, data) = outputs["encoded"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let shape_dims = shape.as_ref();
        let b = shape_dims[0] as usize;
        let d = shape_dims[1] as usize;
        let t = shape_dims[2] as usize;

        let encoder_out = Array3::from_shape_vec((b, d, t), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to reshape encoder output: {e}")))?;

        // on here we are extracting encoded length and new cache states.. and so on...
        let (_, enc_len_data) = outputs["encoded_len"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoded_len: {e}")))?;
        let encoded_len = enc_len_data[0];

        let (ch_shape, ch_data) = outputs["cache_last_channel_next"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_channel: {e}")))?;

        let (tm_shape, tm_data) = outputs["cache_last_time_next"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_time: {e}")))?;

        let (len_shape, len_data) = outputs["cache_last_channel_len_next"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_len: {e}")))?;

        let new_cache = NemotronEncoderCache {
            cache_last_channel: Array4::from_shape_vec(
                (
                    ch_shape[0] as usize,
                    ch_shape[1] as usize,
                    ch_shape[2] as usize,
                    ch_shape[3] as usize,
                ),
                ch_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_last_channel: {e}")))?,

            cache_last_time: Array4::from_shape_vec(
                (
                    tm_shape[0] as usize,
                    tm_shape[1] as usize,
                    tm_shape[2] as usize,
                    tm_shape[3] as usize,
                ),
                tm_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_last_time: {e}")))?,

            cache_last_channel_len: Array1::from_shape_vec(len_shape[0] as usize, len_data.to_vec())
                .map_err(|e| Error::Model(format!("Failed to reshape cache_len: {e}")))?,
        };

        Ok((encoder_out, encoded_len, new_cache))
    }

    /// Run decoder step.
    /// Returns: (logits [vocab_size], new_state_1, new_state_2)
    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>, // [1, hidden_dim, 1]
        target_token: i32,
        state_1: &Array3<f32>, // [2, 1, 640]
        state_2: &Array3<f32>, // [2, 1, 640]
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        let targets = Array2::from_shape_vec((1, 1), vec![target_token])
            .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;
        let target_len = Array1::from_vec(vec![1i32]);

        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => ort::value::Value::from_array(encoder_frame.clone())?,
            "targets" => ort::value::Value::from_array(targets)?,
            "target_length" => ort::value::Value::from_array(target_len)?,
            "input_states_1" => ort::value::Value::from_array(state_1.clone())?,
            "input_states_2" => ort::value::Value::from_array(state_2.clone())?
        ])?;

        // logits for others I think you can understand by looking at the error msgs right? 
        let (_l_shape, l_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        let logits = Array1::from_vec(l_data.to_vec());

        let (h_shape, h_data) = outputs["output_states_1"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state_1: {e}")))?;

        let (c_shape, c_data) = outputs["output_states_2"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state_2: {e}")))?;

        let new_state_1 = Array3::from_shape_vec(
            (
                h_shape[0] as usize,
                h_shape[1] as usize,
                h_shape[2] as usize,
            ),
            h_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape state_1: {e}")))?;

        let new_state_2 = Array3::from_shape_vec(
            (
                c_shape[0] as usize,
                c_shape[1] as usize,
                c_shape[2] as usize,
            ),
            c_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape state_2: {e}")))?;

        Ok((logits, new_state_1, new_state_2))
    }
}
