//! Parakeet Unified (offline + buffered streaming RNN-T) on burn.

use super::generated::{
    unified_decoder_joint, unified_decoder_joint_weights, unified_encoder, unified_encoder_weights,
};
use super::{array3, guard, ints, onnx, tensor3, vec_f32, vec_i64};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array1, Array3, ArrayView3};
use std::path::Path;

pub(crate) struct UnifiedEncoder {
    model: unified_encoder::Model,
    device: Device,
}

impl UnifiedEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut model = guard("allocating the model", || {
            unified_encoder::Model::new(&device)
        })?;
        file.load(&mut model, unified_encoder_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// `features` is `[1, n_mels, time]`; returns the encoder output `[1, dim, frames]` and the
    /// number of valid frames.
    pub(crate) fn run(&self, features: ArrayView3<f32>) -> Result<(Array3<f32>, i64)> {
        let time = features.dim().2;
        let (out, len) = guard("running the Unified encoder", || {
            let (out, len) = self.model.forward(
                tensor3(features, &self.device),
                ints(vec![time as i64], [1], &self.device),
            );
            (array3(out), vec_i64(len))
        })?;
        let len = *len?
            .first()
            .ok_or_else(|| Error::Model("empty encoder length".into()))?;
        Ok((out?, len))
    }
}

pub(crate) struct UnifiedDecoderJoint {
    model: unified_decoder_joint::Model,
    device: Device,
}

impl UnifiedDecoderJoint {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut model = guard("allocating the model", || {
            unified_decoder_joint::Model::new(&device)
        })?;
        file.load(&mut model, unified_decoder_joint_weights::WEIGHTS)?;
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
            let (logits, _, s1, s2) = self.model.forward(
                tensor3(frame, d),
                ints(vec![i64::from(token)], [1, 1], d),
                ints(vec![1], [1], d),
                tensor3(state_1, d),
                tensor3(state_2, d),
            );
            Ok((Array1::from_vec(vec_f32(logits)?), array3(s1)?, array3(s2)?))
        })?
    }
}
