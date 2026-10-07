//! Parakeet CTC on burn.

use super::generated::{ctc, ctc_weights};
use super::{array3, guard, ints, onnx, tensor3};
use crate::error::Result;
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array3, ArrayView3};
use std::path::Path;

pub(crate) struct CtcModel {
    model: ctc::Model,
    device: Device,
}

impl CtcModel {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut model = guard("allocating the model", || ctc::Model::new(&device))?;
        file.load(&mut model, ctc_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// `features` is `[1, time, n_mels]`, all frames valid; returns logits `[1, frames, vocab]`.
    pub(crate) fn run(&self, features: ArrayView3<f32>) -> Result<Array3<f32>> {
        let (b, t, _) = features.dim();
        guard("running the CTC model", || {
            let logits = self.model.forward(
                tensor3(features, &self.device),
                ints(vec![1; b * t], [b, t], &self.device),
            );
            array3(logits)
        })?
    }
}
