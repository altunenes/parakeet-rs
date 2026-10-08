//! Cohere Transcribe on burn. The merged decoder runs as two graphs: the first step (the prompt,
//! which also computes the cross-attention cache) and the later one-token steps.

use super::generated::{
    cohere_decoder_first, cohere_decoder_first_weights, cohere_decoder_next,
    cohere_decoder_next_weights, cohere_encoder, cohere_encoder_weights,
};
use super::{array3, array4, guard, ints, onnx, tensor3, tensor4, vec_f32};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use crate::model_cohere::CoherePastKv;
use burn::tensor::{Device, Tensor};
use ndarray::{Array3, ArrayView3};
use std::path::Path;

pub(crate) struct CohereEncoder {
    model: cohere_encoder::Model,
    device: Device,
}

impl CohereEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut model = guard("allocating the model", || {
            cohere_encoder::Model::new(&device)
        })?;
        file.load(&mut model, cohere_encoder_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// `features` is `[1, time, n_mels]`; returns the hidden states `[1, frames, hidden]`.
    pub(crate) fn run(&self, features: ArrayView3<f32>) -> Result<Array3<f32>> {
        guard("running the encoder", || {
            array3(self.model.forward(tensor3(features, &self.device)))
        })?
    }
}

pub(crate) struct CohereDecoder {
    first: cohere_decoder_first::Model,
    next: cohere_decoder_next::Model,
    device: Device,
}

/// Logits of the last position, and the new cache.
type Step = (Vec<f32>, CoherePastKv);

impl CohereDecoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let mut first = guard("allocating the model", || {
            cohere_decoder_first::Model::new(&device)
        })?;
        file.load(&mut first, cohere_decoder_first_weights::WEIGHTS)?;
        // The later steps use a subset of the first step's weights: share them.
        let next = guard("allocating the model", || {
            cohere_decoder_next::Model::new(&device)
        })?;
        let next = guard("sharing the decoder weights", || {
            onnx::share_weights(
                &first,
                cohere_decoder_first_weights::WEIGHTS,
                next,
                cohere_decoder_next_weights::WEIGHTS,
            )
        })??;
        Ok(Self {
            first,
            next,
            device,
        })
    }

    /// The prompt `tokens`, with an empty cache.
    pub(crate) fn first_step(&self, tokens: &[i64], encoder: ArrayView3<f32>) -> Result<Step> {
        let d = &self.device;
        let n = tokens.len();
        guard("running the decoder", || {
            let (logits, rest) = split_first(self.first.forward(
                ints(tokens.to_vec(), [1, n], d),
                ints(vec![1; n], [1, n], d),
                ints((0..n as i64).collect(), [1, n], d),
                1,
                tensor3(encoder, d),
            ));
            let mut cache = CoherePastKv::empty();
            for (layer, [dk, dv, ek, ev]) in rest.into_iter().enumerate() {
                cache.decoder_k[layer] = array4(dk)?;
                cache.decoder_v[layer] = array4(dv)?;
                cache.encoder_k[layer] = array4(ek)?;
                cache.encoder_v[layer] = array4(ev)?;
            }
            Ok((vec_f32(logits)?, cache))
        })?
    }

    /// One more token after `past`. The graph has no causal mask, so it takes exactly one token.
    pub(crate) fn next_step(&self, token: i64, past: &CoherePastKv) -> Result<Step> {
        let d = &self.device;
        let past_len = past.past_decoder_len();
        if past_len == 0 {
            return Err(Error::Model(
                "Cohere decoder: a later step needs the first step's cache".into(),
            ));
        }
        let cache = |layer: usize| {
            [
                tensor4(past.decoder_k[layer].view(), d),
                tensor4(past.decoder_v[layer].view(), d),
                tensor4(past.encoder_k[layer].view(), d),
                tensor4(past.encoder_v[layer].view(), d),
            ]
        };
        guard("running the decoder", || {
            let [k0, v0, ek0, ev0] = cache(0);
            let [k1, v1, ek1, ev1] = cache(1);
            let [k2, v2, ek2, ev2] = cache(2);
            let [k3, v3, ek3, ev3] = cache(3);
            let [k4, v4, ek4, ev4] = cache(4);
            let [k5, v5, ek5, ev5] = cache(5);
            let [k6, v6, ek6, ev6] = cache(6);
            let [k7, v7, ek7, ev7] = cache(7);
            let (logits, k0, v0, k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, k6, v6, k7, v7) =
                self.next.forward(
                    ints(vec![token], [1, 1], d),
                    ints(vec![1; past_len + 1], [1, past_len + 1], d),
                    ints(vec![past_len as i64], [1, 1], d),
                    1,
                    k0,
                    v0,
                    ek0,
                    ev0,
                    k1,
                    v1,
                    ek1,
                    ev1,
                    k2,
                    v2,
                    ek2,
                    ev2,
                    k3,
                    v3,
                    ek3,
                    ev3,
                    k4,
                    v4,
                    ek4,
                    ev4,
                    k5,
                    v5,
                    ek5,
                    ev5,
                    k6,
                    v6,
                    ek6,
                    ev6,
                    k7,
                    v7,
                    ek7,
                    ev7,
                );
            let mut cache = CoherePastKv {
                decoder_k: Vec::with_capacity(8),
                decoder_v: Vec::with_capacity(8),
                encoder_k: past.encoder_k.clone(),
                encoder_v: past.encoder_v.clone(),
            };
            for (k, v) in [
                (k0, v0),
                (k1, v1),
                (k2, v2),
                (k3, v3),
                (k4, v4),
                (k5, v5),
                (k6, v6),
                (k7, v7),
            ] {
                cache.decoder_k.push(array4(k)?);
                cache.decoder_v.push(array4(v)?);
            }
            Ok((vec_f32(logits)?, cache))
        })?
    }
}

/// The first step's outputs: logits, then per layer the decoder key and value and the encoder
/// key and value.
#[allow(clippy::type_complexity)]
fn split_first(
    out: (
        Tensor<3>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
    ),
) -> (Tensor<3>, Vec<[Tensor<4>; 4]>) {
    let (
        logits,
        a0,
        a1,
        a2,
        a3,
        b0,
        b1,
        b2,
        b3,
        c0,
        c1,
        c2,
        c3,
        d0,
        d1,
        d2,
        d3,
        e0,
        e1,
        e2,
        e3,
        f0,
        f1,
        f2,
        f3,
        g0,
        g1,
        g2,
        g3,
        h0,
        h1,
        h2,
        h3,
    ) = out;
    let layers = vec![
        [a0, a1, a2, a3],
        [b0, b1, b2, b3],
        [c0, c1, c2, c3],
        [d0, d1, d2, d3],
        [e0, e1, e2, e3],
        [f0, f1, f2, f3],
        [g0, g1, g2, g3],
        [h0, h1, h2, h3],
    ];
    (logits, layers)
}
