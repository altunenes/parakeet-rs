use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::tensor_utils::{
    extract_1d_i64, extract_3d_f32, extract_4d_f32, extract_flat_f32, extract_scalar_i64,
};
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{Session, SessionInputValue};
use ort::value::ValueType;
use std::borrow::Cow;
use std::path::Path;

/// Encoder cache state for Nemotron streaming inference.
/// Shapes are model-dependent (English 0.6B uses left_context=70,
/// multilingual 3.5 uses left_context=56) so always construct via [`NemotronEncoderCache::with_dims`].
#[derive(Clone)]
pub struct NemotronEncoderCache {
    pub cache_last_channel: Array4<f32>,
    pub cache_last_time: Array4<f32>,
    pub cache_last_channel_len: Array1<i64>,
}

impl NemotronEncoderCache {
    pub fn with_dims(
        num_layers: usize,
        left_context: usize,
        hidden_dim: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            cache_last_channel: Array4::zeros((num_layers, 1, left_context, hidden_dim)),
            cache_last_time: Array4::zeros((num_layers, 1, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

/// Nemotron ONNX wrapper.
/// Encoder and decoder_joint sessions live side by side; [`Self::has_prompt`]
/// flips on automatically when the encoder graph exposes a `prompt_index` input
/// (the multilingual variant).
pub struct NemotronModel {
    encoder: Session,
    decoder_joint: Session,
    pub config: NemotronModelConfig,
    pub has_prompt: bool,
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
    /// Encoder output frames per chunk (right_context + 1). Read from ONNX
    /// metadata when present, otherwise 7 (the 560ms profile).
    pub chunk_size_output_frames: usize,
    /// Mel frames of pre encode cache per chunk. From ONNX metadata, default 9.
    pub pre_encode_cache: usize,
}

impl NemotronModel {
    /// Load encoder + decoder/joint sessions and read all dimension info
    /// straight from the encoder graph. `vocab_size` is supplied by the
    /// caller (it comes from the tokenizer).
    ///
    /// Note that, multilang graph is identified by the presence of a
    /// `prompt_index` input that flips [`Self::has_prompt`] on.
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        vocab_size: usize,
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

        let encoder = exec_config.build_session(&encoder_path)?;
        let decoder_joint = exec_config.build_session(&decoder_path)?;

        let mut config = NemotronModelConfig {
            num_encoder_layers: 24,
            hidden_dim: 1024,
            left_context: 70,
            conv_context: 8,
            decoder_lstm_dim: 640,
            decoder_lstm_layers: 2,
            vocab_size,
            blank_id: vocab_size,
            chunk_size_output_frames: 7,
            pre_encode_cache: 9,
        };

        // Latency profile from ONNX metadata
        if let Ok(meta) = encoder.metadata() {
            if let Some(v) = meta
                .custom("chunk_size_output_frames")
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
            {
                config.chunk_size_output_frames = v;
            }
            if let Some(v) = meta
                .custom("pre_encode_cache")
                .and_then(|v| v.parse::<usize>().ok())
            {
                config.pre_encode_cache = v;
            }
        }

        let mut has_prompt = false;
        for outlet in encoder.inputs() {
            let name = outlet.name();
            if name == "prompt_index" {
                has_prompt = true;
                continue;
            }
            let ValueType::Tensor { shape, .. } = outlet.dtype() else { continue };
            let dims: &[i64] = shape;
            match name {
                "cache_last_channel" if dims.len() == 4 => {
                    config.num_encoder_layers = dims[0] as usize;
                    config.left_context = dims[2] as usize;
                    config.hidden_dim = dims[3] as usize;
                }
                "cache_last_time" if dims.len() == 4 => {
                    config.conv_context = dims[3] as usize;
                }
                _ => {}
            }
        }

        Ok(Self {
            encoder,
            decoder_joint,
            config,
            has_prompt,
        })
    }

    /// Run encoder with cache-aware streaming.
    /// `prompt_index` must be `Some(_)` for multilingual models and `None`
    /// for eng only mistmaching will produce an ORT InvalidArgument err.
    pub fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &NemotronEncoderCache,
        prompt_index: Option<i64>,
    ) -> Result<(Array3<f32>, i64, NemotronEncoderCache)> {
        let length_arr = Array1::from_vec(vec![length]);

        let mut inputs = ort::inputs![
            "processed_signal" => ort::value::Value::from_array(features.clone())?,
            "processed_signal_length" => ort::value::Value::from_array(length_arr)?,
            "cache_last_channel" => ort::value::Value::from_array(cache.cache_last_channel.clone())?,
            "cache_last_time" => ort::value::Value::from_array(cache.cache_last_time.clone())?,
            "cache_last_channel_len" => ort::value::Value::from_array(cache.cache_last_channel_len.clone())?
        ];
        if let Some(idx) = prompt_index {
            let prompt_arr = Array1::from_vec(vec![idx]);
            inputs.push((
                Cow::Borrowed("prompt_index"),
                SessionInputValue::from(ort::value::Value::from_array(prompt_arr)?),
            ));
        }

        let outputs = self.encoder.run(inputs)?;

        // [1, hidden_dim, time]
        let encoder_out = extract_3d_f32(&outputs["encoded"], "encoder output")?;
        let encoded_len = extract_scalar_i64(&outputs["encoded_len"], "encoded_len")?;

        let new_cache = NemotronEncoderCache {
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

        let logits = extract_flat_f32(&outputs["outputs"], "logits")?;
        let new_state_1 = extract_3d_f32(&outputs["output_states_1"], "state_1")?;
        let new_state_2 = extract_3d_f32(&outputs["output_states_2"], "state_2")?;

        Ok((logits, new_state_1, new_state_2))
    }
}
