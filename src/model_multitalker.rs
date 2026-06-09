use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::tensor_utils::{
    extract_1d_i64, extract_3d_f32, extract_4d_f32, extract_flat_f32, extract_scalar_i64,
};
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

        let encoder = exec_config.build_session(&encoder_path)?;
        let decoder_joint = exec_config.build_session(&decoder_path)?;

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

        let encoder_out = extract_3d_f32(&outputs["encoded"], "encoder output")?;
        let encoded_len = extract_scalar_i64(&outputs["encoded_len"], "encoded_len")?;

        let new_cache = MultitalkerEncoderCache {
            cache_last_channel: extract_4d_f32(
                &outputs["cache_last_channel_next"],
                "cache_last_channel",
            )?,
            cache_last_time: extract_4d_f32(&outputs["cache_last_time_next"], "cache_last_time")?,
            cache_last_channel_len: extract_1d_i64(
                &outputs["cache_last_channel_len_next"],
                "cache_len",
            )?,
        };

        Ok((encoder_out, encoded_len, new_cache))
    }

    /// Run RNNT decoder step.
    ///
    /// The ONNX layout differs from the standard NeMo export (model_nemotron.rs):
    /// encoder_outputs is [B, T, D] (not [B, D, T]), there is no target_length
    /// input, and states are named states_1/states_2. This matches the custom
    /// DecoderJointExport wrapper used in export_multitalker.py.
    ///
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

        let logits = extract_flat_f32(&outputs["outputs"], "logits")?;
        let new_state_1 = extract_3d_f32(&outputs["states_1"], "state_1")?;
        let new_state_2 = extract_3d_f32(&outputs["states_2"], "state_2")?;

        Ok((logits, new_state_1, new_state_2))
    }
}
