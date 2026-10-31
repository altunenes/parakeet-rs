//! Parakeet Unified (offline + buffered streaming RNN-T) on burn, English and multilingual.

use super::generated::{
    unified_decoder_joint, unified_decoder_joint_weights, unified_encoder, unified_encoder_weights,
    unified_multi_decoder_joint, unified_multi_decoder_joint_weights, unified_multi_encoder,
    unified_multi_encoder_weights,
};
use super::{array3, guard, ints, onnx, tensor3, vec_f32, vec_i64};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array1, Array3, ArrayView3};
use std::path::Path;

/// The multilingual encoder takes a `prompt_index` (language) input, and its decoder has a
/// bigger vocabulary, so both are separate graphs.
enum Encoder {
    English(Box<unified_encoder::Model>),
    Multilingual(Box<unified_multi_encoder::Model>),
}

enum DecoderJoint {
    English(Box<unified_decoder_joint::Model>),
    Multilingual(Box<unified_multi_decoder_joint::Model>),
}

pub(crate) struct UnifiedEncoder {
    model: Encoder,
    device: Device,
}

impl UnifiedEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let multilingual = file.info.inputs.iter().any(|(name, _)| name == "prompt_index");
        let model = if multilingual {
            let mut model = guard("allocating the model", || {
                unified_multi_encoder::Model::new(&device)
            })?;
            file.load(&mut model, unified_multi_encoder_weights::WEIGHTS)?;
            Encoder::Multilingual(Box::new(model))
        } else {
            let mut model = guard("allocating the model", || {
                unified_encoder::Model::new(&device)
            })?;
            file.load(&mut model, unified_encoder_weights::WEIGHTS)?;
            Encoder::English(Box::new(model))
        };
        Ok(Self { model, device })
    }

    pub(crate) fn has_prompt(&self) -> bool {
        matches!(self.model, Encoder::Multilingual(_))
    }

    /// `features` is `[1, n_mels, time]`; returns the encoder output `[1, dim, frames]` and the
    /// number of valid frames. `prompt` is the language id, ignored by the English model.
    pub(crate) fn run(&self, features: ArrayView3<f32>, prompt: i64) -> Result<(Array3<f32>, i64)> {
        let time = features.dim().2;
        let (out, len) = guard("running the Unified encoder", || {
            let d = &self.device;
            let (out, len) = match &self.model {
                Encoder::English(m) => {
                    m.forward(tensor3(features, d), ints(vec![time as i64], [1], d))
                }
                Encoder::Multilingual(m) => m.forward(
                    tensor3(features, d),
                    ints(vec![time as i64], [1], d),
                    ints(vec![prompt], [1], d),
                ),
            };
            (array3(out), vec_i64(len))
        })?;
        let len = *len?
            .first()
            .ok_or_else(|| Error::Model("empty encoder length".into()))?;
        Ok((out?, len))
    }
}

pub(crate) struct UnifiedDecoderJoint {
    model: DecoderJoint,
    device: Device,
}

impl UnifiedDecoderJoint {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider, multilingual: bool) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let model = if multilingual {
            let mut model = guard("allocating the model", || {
                unified_multi_decoder_joint::Model::new(&device)
            })?;
            file.load(&mut model, unified_multi_decoder_joint_weights::WEIGHTS)?;
            DecoderJoint::Multilingual(Box::new(model))
        } else {
            let mut model = guard("allocating the model", || {
                unified_decoder_joint::Model::new(&device)
            })?;
            file.load(&mut model, unified_decoder_joint_weights::WEIGHTS)?;
            DecoderJoint::English(Box::new(model))
        };
        Ok(Self { model, device })
    }

    /// One decoder/joint step: flat logits and the new LSTM states.
    pub(crate) fn step(
        &self,
        frame: ArrayView3<f32>,
        token: i32,
        state_1: ArrayView3<f32>,
        state_2: ArrayView3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        guard("running the Unified decoder/joint", || {
            let d = &self.device;
            let args = (
                tensor3(frame, d),
                ints(vec![i64::from(token)], [1, 1], d),
                ints(vec![1], [1], d),
                tensor3(state_1, d),
                tensor3(state_2, d),
            );
            let (logits, _, s1, s2) = match &self.model {
                DecoderJoint::English(m) => m.forward(args.0, args.1, args.2, args.3, args.4),
                DecoderJoint::Multilingual(m) => m.forward(args.0, args.1, args.2, args.3, args.4),
            };
            Ok((Array1::from_vec(vec_f32(logits)?), array3(s1)?, array3(s2)?))
        })?
    }
}
