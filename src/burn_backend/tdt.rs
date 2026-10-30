//! Parakeet TDT encoder and decoder/joint on burn.

#[cfg(feature = "sortformer")]
use super::generated::{multitalker_tdt_encoder, multitalker_tdt_encoder_weights};
use super::generated::{
    tdt_decoder_joint, tdt_decoder_joint_weights, tdt_encoder, tdt_encoder_weights,
};
use super::{array3, guard, ints, onnx, tensor3, vec_f32, vec_i64};
use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use ndarray::{Array3, ArrayView3};
use std::path::Path;

enum EncoderModel {
    Plain(Box<tdt_encoder::Model>),
    #[cfg(feature = "sortformer")]
    Multitalker(Box<multitalker_tdt_encoder::Model>),
}

pub(crate) struct TdtEncoder {
    model: EncoderModel,
    device: Device,
}

impl TdtEncoder {
    pub(crate) fn load(path: &Path, provider: ExecutionProvider) -> Result<Self> {
        let device = super::device(provider)?;
        let file = onnx::OnnxFile::open(path)?;
        let multitalker = file
            .info
            .inputs
            .iter()
            .any(|(name, _)| name == "spk_targets");
        let model = if multitalker {
            load_multitalker(&file, &device)?
        } else {
            let mut model = guard("allocating the model", || tdt_encoder::Model::new(&device))?;
            file.load(&mut model, tdt_encoder_weights::WEIGHTS)?;
            EncoderModel::Plain(Box::new(model))
        };
        Ok(Self { model, device })
    }

    /// `features` is `[1, n_mels, time]`; returns the encoder output `[1, dim, frames]` and the
    /// number of valid frames. `speakers`: (speaker, everyone else) activity, multitalker only.
    pub(crate) fn run(
        &self,
        features: Array3<f32>,
        speakers: Option<&(Vec<f32>, Vec<f32>)>,
    ) -> Result<(Array3<f32>, i64)> {
        let time = features.dim().2;
        let audio = tensor3(features.view(), &self.device);
        let length = ints(vec![time as i64], [1], &self.device);
        let (out, len) = match &self.model {
            EncoderModel::Plain(model) => {
                if speakers.is_some() {
                    return Err(Error::Config(
                        "speaker targets need a multitalker TDT encoder (spk_targets input)".into(),
                    ));
                }
                guard("running the encoder", || {
                    let (out, len) = model.forward(audio, length);
                    (array3(out), vec_i64(len))
                })?
            }
            #[cfg(feature = "sortformer")]
            EncoderModel::Multitalker(model) => {
                let frames = (time - 1) / 8 + 1;
                let (spk, bg) = match speakers {
                    Some((spk, bg)) => (stretch(spk, frames), stretch(bg, frames)),
                    None => (vec![1.0; frames], vec![0.0; frames]),
                };
                let mask = |m: Vec<f32>| {
                    burn::tensor::Tensor::<2>::from_data(
                        burn::tensor::TensorData::new(m, [1, frames]),
                        &self.device,
                    )
                };
                let (spk, bg) = (mask(spk), mask(bg));
                guard("running the encoder", || {
                    let (out, len) = model.forward(audio, length, spk, bg);
                    (array3(out), vec_i64(len))
                })?
            }
        };
        let len = *len?
            .first()
            .ok_or_else(|| Error::Model("empty encoder length".into()))?;
        Ok((out?, len))
    }
}

#[cfg(feature = "sortformer")]
fn load_multitalker(file: &onnx::OnnxFile, device: &Device) -> Result<EncoderModel> {
    let mut model = guard("allocating the model", || {
        multitalker_tdt_encoder::Model::new(device)
    })?;
    file.load(&mut model, multitalker_tdt_encoder_weights::WEIGHTS)?;
    Ok(EncoderModel::Multitalker(Box::new(model)))
}

#[cfg(not(feature = "sortformer"))]
fn load_multitalker(_: &onnx::OnnxFile, _: &Device) -> Result<EncoderModel> {
    Err(Error::Config(
        "multitalker TDT on burn needs the `sortformer` feature".into(),
    ))
}

/// `mask` stretched to `frames` values, nearest earlier frame (what the ONNX export's Resize does).
#[cfg(feature = "sortformer")]
fn stretch(mask: &[f32], frames: usize) -> Vec<f32> {
    if mask.is_empty() {
        return vec![0.0; frames];
    }
    (0..frames).map(|i| mask[i * mask.len() / frames]).collect()
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

#[cfg(all(test, feature = "sortformer"))]
mod stretch_tests {
    use super::stretch;

    #[test]
    fn nearest_earlier_frame() {
        assert_eq!(stretch(&[1.0, 0.0], 4), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(stretch(&[1.0], 3), [1.0, 1.0, 1.0]);
        assert_eq!(stretch(&[1.0, 0.0, 1.0, 0.0], 2), [1.0, 1.0]);
        assert_eq!(stretch(&[], 2), [0.0, 0.0]);
    }
}
