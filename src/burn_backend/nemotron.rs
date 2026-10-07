//! Nemotron cache-aware streaming ASR (English and multilingual) on burn.

use super::generated::{
    nemotron_decoder_joint, nemotron_decoder_joint_weights, nemotron_encoder,
    nemotron_encoder_weights, nemotron_multi_decoder_joint, nemotron_multi_decoder_joint_weights,
    nemotron_multi_encoder, nemotron_multi_encoder_weights,
};
use super::{array3, array4, guard, ints, onnx, tensor3, tensor4, vec_f32, vec_i64};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array1, Array3, Array4, ArrayView3, ArrayView4};
use std::path::Path;

/// The English and the multilingual export are different graphs: the multilingual encoder takes
/// a `prompt_index` (target language) input.
enum Encoder {
    English(Box<nemotron_encoder::Model>),
    Multilingual(Box<nemotron_multi_encoder::Model>),
}

enum DecoderJoint {
    English(Box<nemotron_decoder_joint::Model>),
    Multilingual(Box<nemotron_multi_decoder_joint::Model>),
}

pub(crate) struct NemotronEncoder {
    model: Encoder,
    device: Device,
}

/// One streaming encoder step.
pub(crate) struct EncoderStep {
    /// `[1, hidden, frames]`
    pub encoded: Array3<f32>,
    pub encoded_len: i64,
    pub cache_last_channel: Array4<f32>,
    pub cache_last_time: Array4<f32>,
    pub cache_last_channel_len: Array1<i64>,
}

impl NemotronEncoder {
    /// `multilingual` selects the graph with a `prompt_index` input.
    pub(crate) fn load(
        path: &Path,
        provider: ExecutionProvider,
        multilingual: bool,
    ) -> Result<Self> {
        let device = super::device(provider)?;
        let model = if multilingual {
            let mut model = nemotron_multi_encoder::Model::new(&device);
            onnx::load(&mut model, path, nemotron_multi_encoder_weights::WEIGHTS)?;
            Encoder::Multilingual(Box::new(model))
        } else {
            let mut model = nemotron_encoder::Model::new(&device);
            onnx::load(&mut model, path, nemotron_encoder_weights::WEIGHTS)?;
            Encoder::English(Box::new(model))
        };
        Ok(Self { model, device })
    }

    pub(crate) fn run(
        &self,
        features: ArrayView3<f32>,
        length: i64,
        cache_last_channel: ArrayView4<f32>,
        cache_last_time: ArrayView4<f32>,
        cache_last_channel_len: &Array1<i64>,
        prompt_index: Option<i64>,
    ) -> Result<EncoderStep> {
        guard("running the Nemotron encoder", || {
            let d = &self.device;
            let signal = tensor3(features, d);
            let signal_len = ints(vec![length], [1], d);
            let channel = tensor4(cache_last_channel, d);
            let time = tensor4(cache_last_time, d);
            let channel_len = ints(
                cache_last_channel_len.to_vec(),
                [cache_last_channel_len.len()],
                d,
            );
            let (encoded, encoded_len, channel, time, channel_len) =
                match (&self.model, prompt_index) {
                    (Encoder::English(m), None) => {
                        m.forward(signal, signal_len, channel, time, channel_len)
                    }
                    (Encoder::Multilingual(m), Some(prompt)) => m.forward(
                        signal,
                        signal_len,
                        channel,
                        time,
                        channel_len,
                        ints(vec![prompt], [1], d),
                    ),
                    (Encoder::English(_), Some(_)) => {
                        return Err(Error::Model(
                            "this Nemotron model is English-only; it takes no language prompt"
                                .into(),
                        ));
                    }
                    (Encoder::Multilingual(_), None) => {
                        return Err(Error::Model(
                            "this multilingual Nemotron model needs a language prompt".into(),
                        ));
                    }
                };
            Ok(EncoderStep {
                encoded: array3(encoded)?,
                encoded_len: *vec_i64(encoded_len)?
                    .first()
                    .ok_or_else(|| Error::Model("empty encoded_len".into()))?,
                cache_last_channel: array4(channel)?,
                cache_last_time: array4(time)?,
                cache_last_channel_len: Array1::from_vec(vec_i64(channel_len)?),
            })
        })?
    }
}

pub(crate) struct NemotronDecoderJoint {
    model: DecoderJoint,
    device: Device,
}

impl NemotronDecoderJoint {
    pub(crate) fn load(
        path: &Path,
        provider: ExecutionProvider,
        multilingual: bool,
    ) -> Result<Self> {
        let device = super::device(provider)?;
        let model = if multilingual {
            let mut model = nemotron_multi_decoder_joint::Model::new(&device);
            onnx::load(
                &mut model,
                path,
                nemotron_multi_decoder_joint_weights::WEIGHTS,
            )?;
            DecoderJoint::Multilingual(Box::new(model))
        } else {
            let mut model = nemotron_decoder_joint::Model::new(&device);
            onnx::load(&mut model, path, nemotron_decoder_joint_weights::WEIGHTS)?;
            DecoderJoint::English(Box::new(model))
        };
        Ok(Self { model, device })
    }

    /// One decoder/joint step: logits and the new LSTM states.
    pub(crate) fn step(
        &self,
        frame: ArrayView3<f32>,
        token: i32,
        state_1: ArrayView3<f32>,
        state_2: ArrayView3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        guard("running the Nemotron decoder/joint", || {
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

