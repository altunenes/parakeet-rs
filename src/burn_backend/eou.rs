//! Parakeet realtime EOU (end-of-utterance) streaming ASR on burn.

use super::generated::{
    eou_decoder_joint, eou_decoder_joint_weights, eou_encoder, eou_encoder_weights,
};
use super::{array3, array4, guard, ints, onnx, tensor3, tensor4, vec_i64};
use crate::error::Result;
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array1, Array3, Array4, ArrayView3, ArrayView4};
use std::path::Path;

pub(crate) struct EouEncoder {
    model: eou_encoder::Model,
    device: Device,
}

/// One streaming encoder step: output `[1, hidden, frames]` and the new caches.
pub(crate) type EncoderStep = (Array3<f32>, Array4<f32>, Array4<f32>, Array1<i64>);

impl EouEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let mut model = eou_encoder::Model::new(&device);
        onnx::load(&mut model, path, eou_encoder_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    pub(crate) fn run(
        &self,
        features: ArrayView3<f32>,
        length: i64,
        cache_last_channel: ArrayView4<f32>,
        cache_last_time: ArrayView4<f32>,
        cache_last_channel_len: &Array1<i64>,
    ) -> Result<EncoderStep> {
        guard("running the EOU encoder", || {
            let d = &self.device;
            let (encoded, _, channel, time, channel_len) = self.model.forward(
                tensor3(features, d),
                ints(vec![length], [1], d),
                tensor4(cache_last_channel, d),
                tensor4(cache_last_time, d),
                ints(
                    cache_last_channel_len.to_vec(),
                    [cache_last_channel_len.len()],
                    d,
                ),
            );
            Ok((
                array3(encoded)?,
                array4(channel)?,
                array4(time)?,
                Array1::from_vec(vec_i64(channel_len)?),
            ))
        })?
    }
}

pub(crate) struct EouDecoderJoint {
    model: eou_decoder_joint::Model,
    device: Device,
}

impl EouDecoderJoint {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let mut model = eou_decoder_joint::Model::new(&device);
        onnx::load(&mut model, path, eou_decoder_joint_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// One decoder/joint step: logits `[1, 1, vocab]` and the new LSTM states.
    pub(crate) fn step(
        &self,
        frame: ArrayView3<f32>,
        token: i32,
        state_h: ArrayView3<f32>,
        state_c: ArrayView3<f32>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        guard("running the EOU decoder/joint", || {
            let d = &self.device;
            let (logits, _, h, c) = self.model.forward(
                tensor3(frame, d),
                ints(vec![i64::from(token)], [1, 1], d),
                ints(vec![1], [1], d),
                tensor3(state_h, d),
                tensor3(state_c, d),
            );
            let [a, b, _, v] = logits.dims();
            Ok((array3(logits.reshape([a, b, v]))?, array3(h)?, array3(c)?))
        })?
    }
}
