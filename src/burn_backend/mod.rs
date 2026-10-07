//! Pure-Rust backend on [burn](https://burn.dev): runs models without ONNX Runtime,
//! on the CPU or on the GPU through wgpu (Metal, Vulkan, DX12).
//!
//! The model code in `generated/` comes from burn-onnx and is checked in, so building
//! parakeet-rs never converts models. Weights are read from the same `.onnx` files ONNX Runtime
//! uses (see [`onnx`]). Regenerate with scripts/generate_burn.py when upgrading burn, then check
//! with scripts/check_burn_parity.sh.

pub(crate) mod ctc;
pub(crate) mod eou;
#[cfg(feature = "multitalker")]
pub(crate) mod multitalker;
pub(crate) mod nemotron;
pub(crate) mod onnx;
#[cfg(feature = "sortformer")]
pub(crate) mod sortformer;
pub(crate) mod tdt;
pub(crate) mod unified;

/// One module per generated file.
macro_rules! generated {
    ($($(#[$attr:meta])* $name:ident),* $(,)?) => {
        $(
            $(#[$attr])*
            pub(crate) mod $name {
                include!(concat!("generated/", stringify!($name), ".rs"));
            }
        )*
    };
}

#[rustfmt::skip]
#[allow(clippy::all, unused, non_snake_case)]
mod generated {
    generated!(
        ctc, ctc_weights,
        eou_encoder, eou_encoder_weights, eou_decoder_joint, eou_decoder_joint_weights,
        nemotron_encoder, nemotron_encoder_weights,
        nemotron_decoder_joint, nemotron_decoder_joint_weights,
        nemotron_multi_encoder, nemotron_multi_encoder_weights,
        nemotron_multi_decoder_joint, nemotron_multi_decoder_joint_weights,
        tdt_encoder, tdt_encoder_weights, tdt_decoder_joint, tdt_decoder_joint_weights,
        unified_encoder, unified_encoder_weights, unified_decoder_joint, unified_decoder_joint_weights,
        #[cfg(feature = "multitalker")] multitalker_encoder,
        #[cfg(feature = "multitalker")] multitalker_encoder_weights,
        #[cfg(feature = "multitalker")] multitalker_decoder_joint,
        #[cfg(feature = "multitalker")] multitalker_decoder_joint_weights,
        #[cfg(feature = "sortformer")] sortformer,
        #[cfg(feature = "sortformer")] sortformer_weights,
    );
}

use crate::error::{Error, Result};
use crate::execution::ExecutionProvider;
use burn::tensor::Device;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The burn device for `provider`, or an error if `provider` is not a burn provider or the
/// device cannot be used (for the GPU providers: no usable GPU or driver).
pub(crate) fn device(provider: ExecutionProvider) -> Result<Device> {
    let device = match provider {
        ExecutionProvider::BurnCpu => Device::flex(),
        #[cfg(feature = "wgpu")]
        ExecutionProvider::BurnWgpu => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
        #[cfg(feature = "burn-cuda")]
        ExecutionProvider::BurnCuda => Device::cuda(0),
        #[cfg(feature = "burn-rocm")]
        ExecutionProvider::BurnRocm => Device::rocm(0),
        #[allow(unreachable_patterns)]
        other => {
            return Err(Error::Config(format!(
                "{other:?} is not a burn execution provider"
            )));
        }
    };
    // Devices start lazily; touch this one now so a missing GPU or driver is an error at load
    // time instead of a failure during transcription.
    let probe = guard("initializing the burn device", || {
        burn::tensor::Tensor::<1>::zeros([1], &device).try_into_data()
    })?;
    probe.map_err(|e| {
        Error::Config(format!(
            "{provider:?} is not usable on this machine (no matching GPU or driver?): {e:?}"
        ))
    })?;
    Ok(device)
}

/// Run `f`, turning a burn panic into an error. burn reports device and shape problems by
/// panicking; parakeet-rs returns them as errors.
pub(crate) fn guard<T>(what: &str, f: impl FnOnce() -> T) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|panic| {
        let msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown error");
        Error::Model(format!("burn backend failed while {what}: {msg}"))
    })
}

// ---- ndarray <-> burn ----

pub(crate) fn tensor3(a: ndarray::ArrayView3<f32>, device: &Device) -> burn::tensor::Tensor<3> {
    let (x, y, z) = a.dim();
    let data = a.iter().copied().collect::<Vec<_>>();
    burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(data, [x, y, z]), device)
}

/// An int64 tensor. ONNX integer inputs are int64; `from_data` alone would convert them to
/// burn's default integer type, which the graphs' int64 constants do not mix with.
pub(crate) fn ints<const D: usize>(
    values: Vec<i64>,
    shape: [usize; D],
    device: &Device,
) -> burn::tensor::Tensor<D, burn::tensor::Int> {
    burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(values, shape),
        (device, burn::tensor::DType::I64),
    )
}

pub(crate) fn vec_f32<const D: usize>(t: burn::tensor::Tensor<D>) -> Result<Vec<f32>> {
    t.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .map_err(|e| Error::Model(format!("reading burn output: {e:?}")))
}

pub(crate) fn vec_i64<const D: usize>(
    t: burn::tensor::Tensor<D, burn::tensor::Int>,
) -> Result<Vec<i64>> {
    t.into_data()
        .convert::<i64>()
        .try_into_vec::<i64>()
        .map_err(|e| Error::Model(format!("reading burn output: {e:?}")))
}

pub(crate) fn tensor4(a: ndarray::ArrayView4<f32>, device: &Device) -> burn::tensor::Tensor<4> {
    let (w, x, y, z) = a.dim();
    let data = a.iter().copied().collect::<Vec<_>>();
    burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(data, [w, x, y, z]), device)
}

pub(crate) fn array4(t: burn::tensor::Tensor<4>) -> Result<ndarray::Array4<f32>> {
    let [w, x, y, z] = t.dims();
    ndarray::Array4::from_shape_vec((w, x, y, z), vec_f32(t)?)
        .map_err(|e| Error::Model(format!("burn output shape: {e}")))
}

pub(crate) fn array3(t: burn::tensor::Tensor<3>) -> Result<ndarray::Array3<f32>> {
    let [x, y, z] = t.dims();
    ndarray::Array3::from_shape_vec((x, y, z), vec_f32(t)?)
        .map_err(|e| Error::Model(format!("burn output shape: {e}")))
}

