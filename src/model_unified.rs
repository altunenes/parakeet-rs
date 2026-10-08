#[cfg(feature = "burn")]
use crate::burn_backend::unified::{UnifiedDecoderJoint, UnifiedEncoder};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
#[cfg(feature = "ort")]
use crate::tensor_utils::{extract_3d_f32, extract_flat_f32, extract_scalar_i64};
use ndarray::{Array1, Array2, Array3};
#[cfg(feature = "ort")]
use ort::session::Session;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct UnifiedModelConfig {
    pub vocab_size: usize,
    pub blank_id: usize,
    pub decoder_lstm_dim: usize,
    pub decoder_lstm_layers: usize,
    pub subsampling_factor: usize,
}

impl Default for UnifiedModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1025,
            blank_id: 1024,
            decoder_lstm_dim: 640,
            decoder_lstm_layers: 2,
            subsampling_factor: 8,
        }
    }
}

pub struct ParakeetUnifiedModel {
    encoder: Encoder,
    decoder_joint: DecoderJoint,
    pub config: UnifiedModelConfig,
}

/// The encoder graph, on whichever backend the execution configuration selects.
enum Encoder {
    #[cfg(feature = "ort")]
    Ort(Session),
    #[cfg(feature = "burn")]
    Burn(Box<UnifiedEncoder>),
}

/// The decoder/joint graph, on whichever backend the execution configuration selects.
enum DecoderJoint {
    #[cfg(feature = "ort")]
    Ort(Session),
    #[cfg(feature = "burn")]
    Burn(Box<UnifiedDecoderJoint>),
}

impl ParakeetUnifiedModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        config: UnifiedModelConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        // burn runs fp32 exports only, so it skips int8 files.
        let int8 = !exec_config.execution_provider.is_burn();
        let encoder_path = Self::find_encoder(model_dir, int8)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir, int8)?;

        #[cfg(feature = "burn")]
        if exec_config.execution_provider.is_burn() {
            let provider = exec_config.execution_provider;
            let encoder = UnifiedEncoder::load(&encoder_path, provider)?;
            // The decoder/joint runs once per token: keep it off the GPU.
            let provider = provider.per_token_provider();
            let decoder = UnifiedDecoderJoint::load(&decoder_joint_path, provider)?;
            return Ok(Self {
                encoder: Encoder::Burn(Box::new(encoder)),
                decoder_joint: DecoderJoint::Burn(Box::new(decoder)),
                config,
            });
        }
        #[cfg(feature = "ort")]
        {
            let encoder = exec_config.build_session(&encoder_path)?;
            let decoder_joint = exec_config.build_session(&decoder_joint_path)?;
            Ok(Self {
                encoder: Encoder::Ort(encoder),
                decoder_joint: DecoderJoint::Ort(decoder_joint),
                config,
            })
        }
        #[cfg(not(feature = "ort"))]
        {
            Err(Error::Config(format!(
                "{:?} needs the `ort` feature",
                exec_config.execution_provider
            )))
        }
    }

    fn find_encoder(dir: &Path, int8: bool) -> Result<PathBuf> {
        let candidates = ["encoder.onnx", "encoder.int8.onnx", "encoder-model.onnx"];
        for candidate in candidates.iter().filter(|c| int8 || !c.contains(".int8.")) {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(Error::Config(format!(
            "No unified encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder_joint(dir: &Path, int8: bool) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint.onnx",
            "decoder_joint.int8.onnx",
            "decoder_joint-model.onnx",
        ];
        for candidate in candidates.iter().filter(|c| int8 || !c.contains(".int8.")) {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(Error::Config(format!(
            "No unified decoder_joint model found in {}",
            dir.display()
        )))
    }

    pub fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        let input = features
            .t()
            .to_shape((1, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to build encoder input: {e}")))?
            .to_owned();

        match &mut self.encoder {
            #[cfg(feature = "ort")]
            Encoder::Ort(session) => {
                let input_length = Array1::from_vec(vec![time_steps as i64]);

                let outputs = session.run(ort::inputs!(
                    "audio_signal" => ort::value::Value::from_array(input)?,
                    "length" => ort::value::Value::from_array(input_length)?
                ))?;

                let encoder_out = extract_3d_f32(&outputs["outputs"], "encoder output")?;
                let encoded_len =
                    extract_scalar_i64(&outputs["encoded_lengths"], "encoder lengths")?;

                Ok((encoder_out, encoded_len))
            }
            #[cfg(feature = "burn")]
            Encoder::Burn(encoder) => encoder.run(input.view()),
        }
    }

    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>,
        target_token: i32,
        state_1: &Array3<f32>,
        state_2: &Array3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        match &mut self.decoder_joint {
            #[cfg(feature = "ort")]
            DecoderJoint::Ort(session) => {
                let targets = Array2::from_elem((1, 1), target_token);
                let target_length = Array1::from_elem(1, 1i32);

                let outputs = session.run(ort::inputs![
                    "encoder_outputs" => ort::value::Value::from_array(encoder_frame.clone())?,
                    "targets" => ort::value::Value::from_array(targets)?,
                    "target_length" => ort::value::Value::from_array(target_length)?,
                    "input_states_1" => ort::value::Value::from_array(state_1.clone())?,
                    "input_states_2" => ort::value::Value::from_array(state_2.clone())?
                ])?;

                let logits = extract_flat_f32(&outputs["outputs"], "logits")?;
                let new_state_1 = extract_3d_f32(&outputs["output_states_1"], "state_1")?;
                let new_state_2 = extract_3d_f32(&outputs["output_states_2"], "state_2")?;

                Ok((logits, new_state_1, new_state_2))
            }
            #[cfg(feature = "burn")]
            DecoderJoint::Burn(decoder) => decoder.step(
                encoder_frame.view(),
                target_token,
                state_1.view(),
                state_2.view(),
            ),
        }
    }
}
