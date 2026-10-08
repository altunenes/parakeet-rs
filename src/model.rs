#[cfg(feature = "burn")]
use crate::burn_backend::ctc::CtcModel;
use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::Array2;
#[cfg(feature = "ort")]
use ort::session::Session;
use std::path::Path;

pub struct ParakeetModel {
    backend: Backend,
    config: ModelConfig,
}

/// The model, on whichever backend its execution configuration selects.
enum Backend {
    #[cfg(feature = "ort")]
    Ort(Session),
    #[cfg(feature = "burn")]
    Burn(Box<CtcModel>),
}

impl ParakeetModel {
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::from_pretrained_with_config(model_path, ExecutionConfig::default())
    }

    pub fn from_pretrained_with_config<P: AsRef<Path>>(
        model_path: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Use default config (hardcoded constants for Parakeet-CTC-0.6b: please see: json files https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main)
        let config = ModelConfig::default();

        #[cfg(feature = "burn")]
        if exec_config.execution_provider.is_burn() {
            let model = CtcModel::load(model_path, exec_config.execution_provider)?;
            return Ok(Self {
                backend: Backend::Burn(Box::new(model)),
                config,
            });
        }
        #[cfg(feature = "ort")]
        {
            let session = exec_config.build_session(model_path)?;
            Ok(Self {
                backend: Backend::Ort(session),
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
    pub fn forward(&mut self, features: Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = 1;
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        let input = features
            .to_shape((batch_size, time_steps, feature_size))
            .map_err(|e| Error::Model(format!("Failed to reshape input: {e}")))?
            .to_owned();

        let (shape_dims, data): (Vec<i64>, Vec<f32>) = match &mut self.backend {
            #[cfg(feature = "ort")]
            Backend::Ort(session) => {
                let attention_mask = Array2::<i64>::ones((batch_size, time_steps));

                let input_value = ort::value::Value::from_array(input)?;
                let attention_mask_value = ort::value::Value::from_array(attention_mask)?;

                let outputs = session.run(ort::inputs!(
                    "input_features" => input_value,
                    "attention_mask" => attention_mask_value
                ))?;

                let logits_value = &outputs["logits"];
                let (shape, data) = logits_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
                (shape.to_vec(), data.to_vec())
            }
            #[cfg(feature = "burn")]
            Backend::Burn(model) => {
                let logits = model.run(input.view())?;
                let shape = logits.shape().iter().map(|&d| d as i64).collect();
                (shape, logits.into_raw_vec_and_offset().0)
            }
        };
        if shape_dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D logits, got shape: {shape_dims:?}"
            )));
        }

        let batch_size = shape_dims[0] as usize;
        let time_steps_out = shape_dims[1] as usize;
        let vocab_size = shape_dims[2] as usize;

        if batch_size != 1 {
            return Err(Error::Model(format!(
                "Expected batch size 1, got {batch_size}"
            )));
        }

        let logits_2d = Array2::from_shape_vec((time_steps_out, vocab_size), data)
            .map_err(|e| Error::Model(format!("Failed to create array: {e}")))?;

        Ok(logits_2d)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn pad_token_id(&self) -> usize {
        self.config.pad_token_id
    }
}
