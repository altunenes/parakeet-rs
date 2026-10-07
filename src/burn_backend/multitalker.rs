//! Multitalker (speaker-conditioned cache-aware streaming ASR) on burn.

use super::generated::{
    multitalker_decoder_joint, multitalker_decoder_joint_weights, multitalker_encoder,
    multitalker_encoder_weights,
};
use super::{array3, array4, guard, ints, onnx, tensor3, tensor4, vec_f32, vec_i64};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array1, Array3, Array4, ArrayView2, ArrayView3, ArrayView4};
use std::path::Path;

pub(crate) struct MultitalkerEncoder {
    model: multitalker_encoder::Model,
    device: Device,
}

/// One streaming encoder step: output `[1, hidden, frames]`, its length, and the new caches.
pub(crate) type EncoderStep = (Array3<f32>, i64, Array4<f32>, Array4<f32>, Array1<i64>);

impl MultitalkerEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let mut model = multitalker_encoder::Model::new(&device);
        onnx::load(&mut model, path, multitalker_encoder_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        features: ArrayView3<f32>,
        length: i64,
        cache_last_channel: ArrayView4<f32>,
        cache_last_time: ArrayView4<f32>,
        cache_last_channel_len: &Array1<i64>,
        spk_targets: ArrayView2<f32>,
        bg_spk_targets: ArrayView2<f32>,
    ) -> Result<EncoderStep> {
        guard("running the Multitalker encoder", || {
            let d = &self.device;
            let (encoded, len, channel, time, channel_len) = self.model.forward(
                tensor3(features, d),
                ints(vec![length], [1], d),
                tensor4(cache_last_channel, d),
                tensor4(cache_last_time, d),
                ints(
                    cache_last_channel_len.to_vec(),
                    [cache_last_channel_len.len()],
                    d,
                ),
                tensor2(spk_targets, d),
                tensor2(bg_spk_targets, d),
            );
            let len = *vec_i64(len)?
                .first()
                .ok_or_else(|| Error::Model("empty encoded_len".into()))?;
            Ok((
                array3(encoded)?,
                len,
                array4(channel)?,
                array4(time)?,
                Array1::from_vec(vec_i64(channel_len)?),
            ))
        })?
    }
}

pub(crate) struct MultitalkerDecoderJoint {
    model: multitalker_decoder_joint::Model,
    device: Device,
}

impl MultitalkerDecoderJoint {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let mut model = multitalker_decoder_joint::Model::new(&device);
        onnx::load(&mut model, path, multitalker_decoder_joint_weights::WEIGHTS)?;
        Ok(Self { model, device })
    }

    /// One decoder/joint step for an encoder frame `[1, 1, hidden]`: logits and new states.
    pub(crate) fn step(
        &self,
        frame: ArrayView3<f32>,
        token: i32,
        state_1: ArrayView3<f32>,
        state_2: ArrayView3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        guard("running the Multitalker decoder/joint", || {
            let d = &self.device;
            let (logits, _, s1, s2) = self.model.forward(
                tensor3(frame, d),
                ints(vec![i64::from(token)], [1, 1], d),
                tensor3(state_1, d),
                tensor3(state_2, d),
            );
            Ok((Array1::from_vec(vec_f32(logits)?), array3(s1)?, array3(s2)?))
        })?
    }
}

fn tensor2(a: ArrayView2<f32>, device: &Device) -> burn::tensor::Tensor<2> {
    let (x, y) = a.dim();
    let data = a.iter().copied().collect::<Vec<_>>();
    burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(data, [x, y]), device)
}
