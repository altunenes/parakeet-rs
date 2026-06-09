use crate::error::{Error, Result};
use ndarray::{Array1, Array3, Array4};
use ort::value::DynValue;

/// Extract a tensor as a flat `Array1<f32>` regardless of its ONNX shape.
/// Used for joint-network logits, which arrive as `[1, 1, 1, vocab]`.
pub(crate) fn extract_flat_f32(value: &DynValue, name: &str) -> Result<Array1<f32>> {
    let (_, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    Ok(Array1::from_vec(data.to_vec()))
}

pub(crate) fn extract_3d_f32(value: &DynValue, name: &str) -> Result<Array3<f32>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    let dims = shape.as_ref();
    if dims.len() != 3 {
        return Err(Error::Model(format!(
            "Expected 3D tensor for {name}, got shape: {dims:?}"
        )));
    }
    Array3::from_shape_vec(
        (dims[0] as usize, dims[1] as usize, dims[2] as usize),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("Failed to reshape {name}: {e}")))
}

pub(crate) fn extract_4d_f32(value: &DynValue, name: &str) -> Result<Array4<f32>> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    let dims = shape.as_ref();
    if dims.len() != 4 {
        return Err(Error::Model(format!(
            "Expected 4D tensor for {name}, got shape: {dims:?}"
        )));
    }
    Array4::from_shape_vec(
        (
            dims[0] as usize,
            dims[1] as usize,
            dims[2] as usize,
            dims[3] as usize,
        ),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("Failed to reshape {name}: {e}")))
}

pub(crate) fn extract_1d_i64(value: &DynValue, name: &str) -> Result<Array1<i64>> {
    let (shape, data) = value
        .try_extract_tensor::<i64>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    let dims = shape.as_ref();
    if dims.len() != 1 {
        return Err(Error::Model(format!(
            "Expected 1D tensor for {name}, got shape: {dims:?}"
        )));
    }
    Ok(Array1::from_vec(data.to_vec()))
}

pub(crate) fn extract_scalar_i64(value: &DynValue, name: &str) -> Result<i64> {
    let (_, data) = value
        .try_extract_tensor::<i64>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    data.first()
        .copied()
        .ok_or_else(|| Error::Model(format!("Empty tensor for {name}")))
}

/// Greedy argmax over f32 logits: index and value of the first maximum
pub(crate) fn argmax_f32(values: impl IntoIterator<Item = f32>) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, v) in values.into_iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    (max_idx, max_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_first_maximum() {
        assert_eq!(argmax_f32([0.1, 0.5, 0.3, 0.9, 0.2]).0, 3);
        assert_eq!(argmax_f32([1.0, 0.0, 1.0]).0, 0);
        assert_eq!(argmax_f32([f32::NAN, 0.5, 0.3]).0, 1);
        assert_eq!(argmax_f32([]), (0, f32::NEG_INFINITY));
    }
}
