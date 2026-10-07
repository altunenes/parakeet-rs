//! Parakeet TDT encoder and decoder/joint on burn.

use super::generated::{
    tdt_decoder_joint, tdt_decoder_joint_weights, tdt_encoder, tdt_encoder_weights,
};
use super::{array3, guard, ints, onnx, tensor3, vec_f32, vec_i64};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array3, ArrayView3};
use std::path::Path;

pub(crate) struct TdtEncoder {
    model: tdt_encoder::Model,
    device: Device,
}

impl TdtEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut model = guard("allocating the model", || tdt_encoder::Model::new(&device))?;
        file.load(&mut model, tdt_encoder_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// `features` is `[1, n_mels, time]`; returns the encoder output `[1, dim, frames]` and the
    /// number of valid frames.
    pub(crate) fn run(&self, features: Array3<f32>) -> Result<(Array3<f32>, i64)> {
        let time = features.dim().2;
        let (out, len) = guard("running the encoder", || {
            let (out, len) = self.model.forward(
                tensor3(features.view(), &self.device),
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

/// Output of one decoder/joint step.
pub(crate) struct JointStep {
    /// Token logits followed by duration logits.
    pub logits: Vec<f32>,
    pub state_h: Array3<f32>,
    pub state_c: Array3<f32>,
}

pub(crate) struct TdtDecoderJoint {
    model: tdt_decoder_joint::Model,
    device: Device,
}

impl TdtDecoderJoint {
    /// `vocab_size` counts the blank token.
    pub(crate) fn load(
        path: &Path,
        provider: ExecutionProvider,
        vocab_size: usize,
    ) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        // The joint emits one logit per token plus one per duration; the duration count is a
        // property of the export, so read the joint's output size from the weights.
        let joint_size = joint_output_size(&file)?;
        if joint_size <= vocab_size {
            return Err(Error::Model(format!(
                "{}: joint output size {joint_size} does not fit vocabulary size {vocab_size}",
                path.display()
            )));
        }
        let mut model = guard("allocating the model", || {
            tdt_decoder_joint::Model::new(&device, vocab_size, joint_size)
        })?;
        file.load(&mut model, tdt_decoder_joint_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// One decoder/joint step for a single encoder frame `[1, dim, 1]`, the previous token, and
    /// the LSTM state `[2, 1, 640]` (h and c).
    pub(crate) fn step(
        &self,
        frame: ArrayView3<f32>,
        token: i32,
        state_h: &Array3<f32>,
        state_c: &Array3<f32>,
    ) -> Result<JointStep> {
        guard("running the decoder/joint", || {
            let device = &self.device;
            let (logits, _, h, c) = self.model.forward(
                tensor3(frame, device),
                ints(vec![i64::from(token)], [1, 1], device),
                ints(vec![1], [1], device),
                tensor3(state_h.view(), device),
                tensor3(state_c.view(), device),
            );
            Ok(JointStep {
                logits: vec_f32(logits)?,
                state_h: array3(h)?,
                state_c: array3(c)?,
            })
        })?
    }
}

/// Output size of the joint network: the length of its last layer's bias.
fn joint_output_size(file: &onnx::OnnxFile) -> Result<usize> {
    let source = tdt_decoder_joint_weights::WEIGHTS
        .iter()
        .find(|w| w.path == "linear3.bias")
        .map(|w| &w.source)
        .ok_or_else(|| Error::Model("generated decoder table has no joint output layer".into()))?;
    let onnx::Source::Param(key, _) = source else {
        return Err(Error::Model(
            "joint output bias is not read from the model".into(),
        ));
    };
    Ok(file.dims(key)?.iter().product())
}
