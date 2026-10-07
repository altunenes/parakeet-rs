//! Sortformer (Nemotron-3 Diarization) on burn.

use super::generated::{sortformer, sortformer_weights};
use super::{array3, guard, ints, onnx, tensor3};
use crate::error::Result;
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array3, ArrayView3};
use std::path::Path;

pub(crate) struct SortformerModel {
    model: sortformer::Model,
    device: Device,
}

/// The outputs Sortformer's streaming update uses.
pub(crate) struct SortformerStep {
    /// 80ms speaker activity over spkcache + fifo + chunk, `[1, frames, speakers]`.
    pub preds_diar: Array3<f32>,
    /// 10ms speaker activity for the chunk, `[1, frames, speakers]`.
    pub preds_hires: Array3<f32>,
    /// The chunk's pre-encoder embeddings, `[1, frames, emb_dim]`.
    pub chunk_embs: Array3<f32>,
}

impl SortformerModel {
    /// Load the model, with the export's output names and metadata.
    pub(crate) fn load(
        path: &Path,
        provider: ExecutionProvider,
    ) -> Result<(Self, onnx::ModelInfo)> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut model = guard("allocating the model", || sortformer::Model::new(&device))?;
        file.load(&mut model, sortformer_weights::WEIGHTS)?;
        Ok((Self { model, device }, file.info))
    }

    /// One streaming step. `chunk` is `[1, mel frames, n_mels]` with `chunk_len` valid frames;
    /// `spkcache` and `fifo` are `[1, frames, emb_dim]` and may have zero frames.
    pub(crate) fn run(
        &self,
        chunk: ArrayView3<f32>,
        chunk_len: usize,
        spkcache: ArrayView3<f32>,
        fifo: ArrayView3<f32>,
    ) -> Result<SortformerStep> {
        // burn cannot reshape zero-length tensors, and every stream starts with an empty speaker
        // cache and FIFO. The graph packs only the first `*_lengths` frames of each input, so a
        // single masked frame with length 0 gives the same result as an empty input.
        let placeholder = Array3::<f32>::zeros((1, 1, spkcache.dim().2));
        let spkcache_in = if spkcache.dim().1 == 0 {
            placeholder.view()
        } else {
            spkcache
        };
        let fifo_in = if fifo.dim().1 == 0 {
            placeholder.view()
        } else {
            fifo
        };
        guard("running Sortformer", || {
            let device = &self.device;
            let (preds_diar, preds_hires, chunk_embs, _) = self.model.forward(
                tensor3(chunk, device),
                ints(vec![chunk_len as i64], [1], device),
                tensor3(spkcache_in, device),
                ints(vec![spkcache.dim().1 as i64], [1], device),
                tensor3(fifo_in, device),
                ints(vec![fifo.dim().1 as i64], [1], device),
            );
            Ok(SortformerStep {
                preds_diar: array3(preds_diar)?,
                preds_hires: array3(preds_hires)?,
                chunk_embs: array3(chunk_embs)?,
            })
        })?
    }
}
