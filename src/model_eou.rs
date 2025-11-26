use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::path::Path;

/// Encoder cache state for streaming inference
/// The cache maintains temporal context across chunks
#[derive(Default)]
pub struct EncoderCache {
    /// channel cache: [1, 1, 70, 512] - batch=1, 70 frame lookback
    pub cache_last_channel: Array4<f32>,
    /// time cache: [1, 1, 512, 8] - batch=1, fixed 8 time steps
    pub cache_last_time: Array4<f32>,
    /// cache length: [1] with value 0 initially
    pub cache_last_channel_len: Array1<i64>,
}

impl EncoderCache {
    /// 17 layers, batch=1, 70 frame lookback, 512 features
    pub fn new() -> Self {
        Self {
            cache_last_channel: Array4::zeros((17, 1, 70, 512)),
            cache_last_time: Array4::zeros((17, 1, 512, 8)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

pub struct ParakeetEOUModel {
    encoder: Session,
    decoder_joint: Session,
}

impl ParakeetEOUModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = model_dir.join("encoder.onnx");
        let decoder_path = model_dir.join("decoder_joint.onnx");

        if !encoder_path.exists() || !decoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing ONNX files in {}. Expected encoder.onnx and decoder_joint.onnx",
                model_dir.display()
            )));
        }

        // Load encoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
        })
    }

    /// Run the stateful encoder with cache
    /// Input: features [1, 128, T], cache state
    /// Output: (encoded [1, 512, T], new_cache)
    pub fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &EncoderCache,
    ) -> Result<(Array3<f32>, EncoderCache)> {
        let length_arr = Array1::from_vec(vec![length]);

        let outputs = self.encoder.run(ort::inputs![
            "audio_signal" => ort::value::Value::from_array(features.clone())?,
            "length" => ort::value::Value::from_array(length_arr)?,
            "cache_last_channel" => ort::value::Value::from_array(cache.cache_last_channel.clone())?,
            "cache_last_time" => ort::value::Value::from_array(cache.cache_last_time.clone())?,
            "cache_last_channel_len" => ort::value::Value::from_array(cache.cache_last_channel_len.clone())?
        ])?;

        // Extract encoder output [1, 512, T]
        let (shape, data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let shape_dims = shape.as_ref();
        let b = shape_dims[0] as usize;
        let d = shape_dims[1] as usize;
        let t = shape_dims[2] as usize;

        let encoder_out = Array3::from_shape_vec((b, d, t), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to reshape encoder output: {e}")))?;

        // Extract new cache states
        let (ch_shape, ch_data) = outputs["new_cache_last_channel"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_channel: {e}")))?;

        let (tm_shape, tm_data) = outputs["new_cache_last_time"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_last_time: {e}")))?;

        let (len_shape, len_data) = outputs["new_cache_last_channel_len"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract cache_len: {e}")))?;

        // Build new cache with extracted shapes
        let new_cache = EncoderCache {
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

            cache_last_channel_len: Array1::from_shape_vec(
                len_shape[0] as usize,
                len_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_len: {e}")))?,
        };

        Ok((encoder_out, new_cache))
    }

    /// Run the stateful decoder
    /// Returns: (logits [1, 1, 1, vocab], new_state_h, new_state_c)
    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>, // [1, 512, 1]
        last_token: &Array2<i32>,    // [1, 1]
        state_h: &Array3<f32>,       // [1, 1, 640]
        state_c: &Array3<f32>,       // [1, 1, 640]
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        // Target length is always 1 for single step
        let target_len = Array1::from_vec(vec![1i32]);

        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => ort::value::Value::from_array(encoder_frame.clone())?,
            "targets" => ort::value::Value::from_array(last_token.clone())?,
            "target_length" => ort::value::Value::from_array(target_len)?,
            "input_states_1" => ort::value::Value::from_array(state_h.clone())?,
            "input_states_2" => ort::value::Value::from_array(state_c.clone())?
        ])?;

        // 1. Extract Logits
        let (l_shape, l_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        // 2. Extract States (output_states_1, output_states_2)
        let (_h_shape, h_data) = outputs["output_states_1"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state h: {e}")))?;

        let (_c_shape, c_data) = outputs["output_states_2"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state c: {e}")))?;

        // Reconstruct Arrays
        // Logits: I simplify to [1, 1, vocab]
        let vocab_size = l_shape[3] as usize;
        let logits = Array3::from_shape_vec((1, 1, vocab_size), l_data.to_vec())
            .map_err(|e| Error::Model(format!("Reshape logits failed: {e}")))?;

        // States: [1, 1, 640]
        let new_h = Array3::from_shape_vec((1, 1, 640), h_data.to_vec())
            .map_err(|e| Error::Model(format!("Reshape state h failed: {e}")))?;

        let new_c = Array3::from_shape_vec((1, 1, 640), c_data.to_vec())
            .map_err(|e| Error::Model(format!("Reshape state c failed: {e}")))?;

        Ok((logits, new_h, new_c))
    }
}
