//! Sortformer (Nemotron-3 Diarization) on burn.

use super::generated::{sortformer, sortformer_weights};
use super::{array3, guard, ints, onnx, tensor3};
use crate::error::Result;
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array3, ArrayView3, Axis};
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
        // burn can't take empty tensors, so split the non-empty cache across both inputs (the
        // graph joins them anyway). With under two cached frames (a stream's start) a masked
        // placeholder frame remains; it slightly shifts the last valid frame.
        let (spkcache, fifo) = split_caches(spkcache, fifo);
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

/// The same frames, spkcache then FIFO, with both non-empty when there are two or more.
fn split_caches<'a>(
    spkcache: ArrayView3<'a, f32>,
    fifo: ArrayView3<'a, f32>,
) -> (ArrayView3<'a, f32>, ArrayView3<'a, f32>) {
    match (spkcache.dim().1, fifo.dim().1) {
        (0, n) if n > 1 => fifo.split_at(Axis(1), 1),
        (n, 0) if n > 1 => spkcache.split_at(Axis(1), n - 1),
        _ => (spkcache, fifo),
    }
}

#[cfg(test)]
mod tests {
    use super::split_caches;
    use ndarray::{Array3, Axis, concatenate};

    #[test]
    fn split_caches_keeps_the_frame_order() {
        let frames = |n: usize, start: usize| {
            Array3::from_shape_fn((1, n, 2), |(_, t, f)| (start + t * 2 + f) as f32)
        };
        for (s, f) in [(0, 0), (0, 1), (1, 0), (0, 3), (3, 0), (2, 2)] {
            let (spkcache, fifo) = (frames(s, 0), frames(f, 100));
            let (a, b) = split_caches(spkcache.view(), fifo.view());
            let joined = |x, y| concatenate(Axis(1), &[x, y]).unwrap();
            assert_eq!(joined(a, b), joined(spkcache.view(), fifo.view()), "({s}, {f})");
            if s + f >= 2 {
                assert!(a.dim().1 > 0 && b.dim().1 > 0, "({s}, {f})");
            }
        }
    }
}
