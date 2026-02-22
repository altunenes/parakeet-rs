use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::path::Path;

/// Encoder cache for the multitalker model.
///
/// Unlike `NemotronEncoderCache` which uses `[n_layers, batch, ...]` ordering,
/// the multitalker ONNX encoder expects `[batch, n_layers, ...]` because the
/// export wrapper calls `forward_for_export()` which transposes (0,1) internally.
#[derive(Clone)]
pub(crate) struct MultitalkerEncoderCache {
    /// [1, n_layers, left_context, d_model] - batch-first cache
    pub(crate) cache_last_channel: Array4<f32>,
    /// [1, n_layers, d_model, conv_context] - batch-first cache
    pub(crate) cache_last_time: Array4<f32>,
    /// [1] - current cache length
    pub(crate) cache_last_channel_len: Array1<i64>,
}

impl MultitalkerEncoderCache {
    pub(crate) fn new(
        num_layers: usize,
        left_context: usize,
        hidden_dim: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            // batch-first: [1, n_layers, left_context, hidden_dim]
            cache_last_channel: Array4::zeros((1, num_layers, left_context, hidden_dim)),
            // batch-first: [1, n_layers, hidden_dim, conv_context]
            cache_last_time: Array4::zeros((1, num_layers, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

/// Multitalker ONNX wrapper.
/// Encoder accepts additional spk_targets and bg_spk_targets inputs for speaker
/// kernel injection. Decoder is identical to Nemotron's RNNT decoder.
pub(crate) struct MultitalkerModel {
    encoder: Session,
    decoder_joint: Session,
}

impl MultitalkerModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Prefer int8 models if available
        let encoder_path = {
            let int8 = model_dir.join("encoder.int8.onnx");
            let fp32 = model_dir.join("encoder.onnx");
            if int8.exists() {
                int8
            } else if fp32.exists() {
                fp32
            } else {
                return Err(Error::Config(format!(
                    "Missing encoder.onnx or encoder.int8.onnx in {}",
                    model_dir.display()
                )));
            }
        };

        let decoder_path = {
            let int8 = model_dir.join("decoder_joint.int8.onnx");
            let fp32 = model_dir.join("decoder_joint.onnx");
            if int8.exists() {
                int8
            } else if fp32.exists() {
                fp32
            } else {
                return Err(Error::Config(format!(
                    "Missing decoder_joint.onnx or decoder_joint.int8.onnx in {}",
                    model_dir.display()
                )));
            }
        };

        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
        })
    }

    /// Run encoder with cache-aware streaming and speaker target injection.
    ///
    /// Compared to NemotronModel::run_encoder(), this adds two extra inputs:
    /// - `spk_targets`: per-frame target speaker activity [1, T_enc]
    /// - `bg_spk_targets`: per-frame background speaker activity [1, T_enc]
    ///
    /// Cache format is batch-first: [1, n_layers, ...] (unlike Nemotron which
    /// uses [n_layers, 1, ...]).
    pub(crate) fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &MultitalkerEncoderCache,
        spk_targets: &Array2<f32>,
        bg_spk_targets: &Array2<f32>,
    ) -> Result<(Array3<f32>, i64, MultitalkerEncoderCache)> {
        let length_arr = Array1::from_vec(vec![length]);

        let outputs = self.encoder.run(ort::inputs![
            "processed_signal" => ort::value::Value::from_array(features.clone())?,
            "processed_signal_length" => ort::value::Value::from_array(length_arr)?,
            "cache_last_channel" => ort::value::Value::from_array(cache.cache_last_channel.clone())?,
            "cache_last_time" => ort::value::Value::from_array(cache.cache_last_time.clone())?,
            "cache_last_channel_len" => ort::value::Value::from_array(cache.cache_last_channel_len.clone())?,
            "spk_targets" => ort::value::Value::from_array(spk_targets.clone())?,
            "bg_spk_targets" => ort::value::Value::from_array(bg_spk_targets.clone())?
        ])?;

        let (shape, data) = outputs["encoded"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let shape_dims = shape.as_ref();
        let b = shape_dims[0] as usize;
        let d = shape_dims[1] as usize;
        let t = shape_dims[2] as usize;

        let encoder_out = Array3::from_shape_vec((b, d, t), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to reshape encoder output: {e}")))?;

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

        let new_cache = MultitalkerEncoderCache {
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

        Ok((encoder_out, encoded_len, new_cache))
    }

    /// Run decoder step (identical to NemotronModel::run_decoder).
    /// Returns: (logits [vocab_size+1], new_state_1, new_state_2)
    pub(crate) fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>,
        target_token: i32,
        state_1: &Array3<f32>,
        state_2: &Array3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        let targets = Array2::from_shape_vec((1, 1), vec![target_token as i64])
            .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => ort::value::Value::from_array(encoder_frame.clone())?,
            "targets" => ort::value::Value::from_array(targets)?,
            "input_states_1" => ort::value::Value::from_array(state_1.clone())?,
            "input_states_2" => ort::value::Value::from_array(state_2.clone())?
        ])?;

        let (_l_shape, l_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        let logits = Array1::from_vec(l_data.to_vec());

        let (h_shape, h_data) = outputs["states_1"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state_1: {e}")))?;

        let (c_shape, c_data) = outputs["states_2"]
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
