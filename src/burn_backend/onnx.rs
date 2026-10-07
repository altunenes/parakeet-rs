//! Load burn model weights straight from `.onnx` files.
//!
//! The burn models are generated from one reference export (see scripts/generate_burn.py),
//! but users have other exports of the same models: newer PyTorch versions name anonymous
//! weights differently, store some layers in another layout, and size buffers differently. So
//! weights are not found by the reference export's tensor names but by a key that holds across
//! exports ([`weight_key`]), with a few known layout differences handled in [`resolve`].
//!
//! Only the parts of the ONNX protobuf that hold tensors are read: graph initializers and
//! `Constant` nodes, plus node names and inputs for the keys. Weights stored as external data
//! are not read here; burn reads them from the data file when it moves them to the device.

use crate::error::{Error, Result};
use burn::tensor::{Bytes, DType};
use burn_store::ModuleSnapshot;
use burn_store::burn_pack::Tensor as PackTensor;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One burn parameter and where its data comes from. Generated tables list these.
pub(crate) struct Weight {
    /// The parameter's path in the burn module.
    pub path: &'static str,
    /// The parameter's shape, or `None` to take it from the ONNX tensor (sizes that depend on
    /// the vocabulary).
    pub shape: Option<&'static [usize]>,
    pub source: Source,
}

pub(crate) enum Source {
    /// A tensor in the user's ONNX file, found by [`weight_key`], then transformed.
    Param(&'static str, Transform),
    /// A small graph constant of the reference export.
    ConstF32(&'static [f32]),
    ConstI64(&'static [i64]),
    ConstI32(&'static [i32]),
    /// All zeros.
    Zeros,
}

/// How a burn parameter is built from its ONNX tensor.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Transform {
    /// The tensor as is.
    Copy,
    /// The centered rows of a `[1, N, D]` table (relative positional encoding). Exports size
    /// the table from the audio length they were traced with; positions are centered on zero,
    /// so a longer table holds the reference one in its middle.
    CenterRows,
    /// Rows `start..end` of a packed `[1, 4H, I]` LSTM weight, transposed to `[I, H]`.
    RowsT(usize, usize),
    /// Sum of two slices of a packed `[1, 8H]` LSTM bias: burn folds the input and recurrent
    /// biases of a gate into one.
    BiasSum(usize, usize, usize, usize),
}

/// The key that finds a weight in any export: an initializer's own name when it has one (NeMo
/// parameter names like `layers.0.norm_ff.weight`), else the nodes consuming it, since
/// anonymous names (`onnx::MatMul_6382`) are numbered by the exporter and differ between
/// exports. Same rule as `weight_key` in scripts/generate_burn.py.
fn weight_key(name: &str, consumers: Option<&Vec<String>>) -> String {
    if !name.starts_with("onnx::") {
        return name.to_string();
    }
    let mut consumers = consumers.cloned().unwrap_or_default();
    consumers.sort();
    consumers.join("|")
}

/// Load every weight in `table` from `onnx_path` into `model`.
///
/// Fails with a clear error, never a panic, if the file is not an export of the model the
/// table was generated for.
pub(crate) fn load<M: ModuleSnapshot>(
    model: &mut M,
    onnx_path: &Path,
    table: &[Weight],
) -> Result<()> {
    let tensors = read_tensors(onnx_path)?;
    let unsupported = |what: String| {
        Error::Model(format!(
            "{}: {what}. This ONNX export is not supported by the burn backend; use ONNX Runtime",
            onnx_path.display()
        ))
    };

    let mut pack = Vec::with_capacity(table.len());
    for w in table {
        let tensor = match &w.source {
            Source::ConstF32(values) => {
                constant(w, DType::F32, Bytes::from_elems(values.to_vec()))?
            }
            Source::ConstI64(values) => {
                constant(w, DType::I64, Bytes::from_elems(values.to_vec()))?
            }
            Source::ConstI32(values) => {
                constant(w, DType::I32, Bytes::from_elems(values.to_vec()))?
            }
            Source::Zeros => {
                let numel = shape_of(w)?.iter().product::<usize>();
                constant(w, DType::F32, Bytes::from_elems(vec![0f32; numel]))?
            }
            Source::Param(key, transform) => {
                let (src, layout) = resolve(&tensors, key)
                    .ok_or_else(|| unsupported(format!("no tensor for {key}")))?;
                param(w, src, layout, *transform).map_err(unsupported)?
            }
        };
        pack.push(tensor);
    }

    let result = model.apply(pack, None, None, false);
    if !result.missing.is_empty() || !result.unused.is_empty() || !result.errors.is_empty() {
        return Err(unsupported(format!(
            "weights did not load cleanly (missing {:?}, unused {:?}, errors {:?})",
            result.missing.iter().take(3).collect::<Vec<_>>(),
            result.unused.iter().take(3).collect::<Vec<_>>(),
            result.errors.iter().take(3).collect::<Vec<_>>()
        )));
    }
    Ok(())
}

/// Shape of the tensor for `key` in `onnx_path`.
pub(crate) fn dims(onnx_path: &Path, key: &str) -> Result<Vec<usize>> {
    let tensors = read_tensors(onnx_path)?;
    resolve(&tensors, key)
        .map(|(t, _)| t.dims.clone())
        .ok_or_else(|| Error::Model(format!("{}: no tensor for {key}", onnx_path.display())))
}

fn constant(w: &Weight, dtype: DType, bytes: Bytes) -> Result<PackTensor> {
    Ok(PackTensor::new(
        w.path.into(),
        dtype,
        shape_of(w)?,
        None,
        bytes,
    ))
}

fn shape_of(w: &Weight) -> Result<Vec<usize>> {
    w.shape
        .map(<[usize]>::to_vec)
        .ok_or_else(|| Error::Model(format!("{}: a constant needs a fixed shape", w.path)))
}

/// How a tensor found under another layout has to be turned into the reference layout.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    Reference,
    /// A `[in, out]` matrix where the reference has `[out, in]` or `[out, in, 1]`.
    Transposed,
}

/// Find the tensor for `key`, also under the layouts other exports use for the same weight.
fn resolve<'a>(
    tensors: &'a HashMap<String, OnnxTensor>,
    key: &str,
) -> Option<(&'a OnnxTensor, Layout)> {
    if let Some(t) = tensors.get(key) {
        return Some((t, Layout::Reference));
    }
    // Newer PyTorch exports a kernel-size-1 Conv1d as MatMul: `layers.N.conv.pointwise_conv1.weight`
    // `[out, in, 1]` becomes the anonymous input of `/layers.N/conv/MatMul` as `[in, out]`
    // (`MatMul_1` for pointwise_conv2).
    if let Some(rest) = key.strip_prefix("layers.")
        && let Some((layer, conv)) = rest.split_once(".conv.pointwise_conv")
        && let Some(n) = conv.strip_suffix(".weight")
    {
        let node = if n == "1" { "MatMul" } else { "MatMul_1" };
        return tensors
            .get(&format!("/layers.{layer}/conv/{node}#1"))
            .map(|t| (t, Layout::Transposed));
    }
    // A linear weight that one export keeps anonymous (`[in, out]`, consumed by `/a/b/MatMul`)
    // can be named in another (`a.b.weight`, `[out, in]`).
    if let Some(module) = key.strip_suffix("/MatMul#1")
        && !key.contains('|')
    {
        let name = format!(
            "{}.weight",
            module.trim_start_matches('/').replace('/', ".")
        );
        return tensors.get(&name).map(|t| (t, Layout::Transposed));
    }
    None
}

fn param(
    w: &Weight,
    src: &OnnxTensor,
    layout: Layout,
    transform: Transform,
) -> std::result::Result<PackTensor, String> {
    let dtype = src.dtype()?;
    let numel = |shape: &[usize]| shape.iter().product::<usize>();
    let target = match (w.shape, transform) {
        (Some(shape), _) => shape.to_vec(),
        (None, Transform::Copy) => src.dims.clone(),
        (None, _) => return Err(format!("{}: missing shape", w.path)),
    };
    let check = |ok: bool| {
        if ok {
            Ok(())
        } else {
            Err(format!(
                "{} has shape {:?}, expected {target:?}",
                w.path, src.dims
            ))
        }
    };

    if let (Transform::Copy, Layout::Reference) = (transform, layout) {
        check(numel(&src.dims) == numel(&target))?;
        // Plain copies (nearly all weights) stay lazy: burn reads them straight from the file.
        return Ok(PackTensor::new(
            w.path.into(),
            dtype,
            target,
            None,
            src.bytes()?,
        ));
    }

    // Everything else is computed when burn asks for the bytes, one tensor at a time.
    let elem = match transform {
        Transform::Copy => {
            check(
                layout == Layout::Transposed
                    && src.dims.len() == 2
                    && numel(&src.dims) == numel(&target),
            )?;
            Elementwise::Transpose
        }
        Transform::CenterRows => {
            let [1, rows, cols] = src.dims[..] else {
                return Err(format!("{}: expected a [1, N, D] table", w.path));
            };
            let [1, want, d] = target[..] else {
                return Err(format!("{}: unexpected reference shape", w.path));
            };
            check(cols == d && rows >= want && (rows - want) % 2 == 0)?;
            Elementwise::Rows((rows - want) / 2 * d..((rows - want) / 2 + want) * d)
        }
        Transform::RowsT(a, b) => {
            check(src.dims.len() == 3 && numel(&target) == (b - a) * src.dims[2])?;
            Elementwise::RowsT(a, b, src.dims[2])
        }
        Transform::BiasSum(a, b, c, d) => {
            check(
                src.dims.len() == 2
                    && b - a == d - c
                    && numel(&target) == b - a
                    && d <= src.dims[1],
            )?;
            Elementwise::Sum(a..b, c..d)
        }
    };
    if dtype != DType::F32 {
        return Err(format!("{}: only float tensors can be transformed", w.path));
    }
    let src = src.clone();
    let len = numel(&target) * 4;
    let rows = src.dims.first().copied().unwrap_or(0);
    let path = w.path;
    Ok(PackTensor::deferred(
        w.path.into(),
        dtype,
        target,
        None,
        len,
        move || {
            let v = src.f32s().map_err(burn_store::burn_pack::Error::IoError)?;
            let out: Vec<f32> = match &elem {
                Elementwise::Transpose => {
                    let cols = v.len() / rows;
                    (0..cols)
                        .flat_map(|c| (0..rows).map(move |r| (r, c)))
                        .map(|(r, c)| v[r * cols + c])
                        .collect()
                }
                Elementwise::Rows(r) => v[r.clone()].to_vec(),
                Elementwise::RowsT(a, b, cols) => (0..*cols)
                    .flat_map(|c| (*a..*b).map(move |r| (r, c)))
                    .map(|(r, c)| v[r * cols + c])
                    .collect(),
                Elementwise::Sum(x, y) => v[x.clone()]
                    .iter()
                    .zip(&v[y.clone()])
                    .map(|(p, q)| p + q)
                    .collect(),
            };
            if out.len() * 4 != len {
                return Err(burn_store::burn_pack::Error::IoError(format!(
                    "{path}: transform produced {} values",
                    out.len()
                )));
            }
            Ok(Bytes::from_elems(out))
        },
    ))
}

enum Elementwise {
    /// Transpose a 2-D matrix.
    Transpose,
    /// A contiguous range of the flat data.
    Rows(Range<usize>),
    /// Rows `a..b` of a `[.., cols]` matrix, transposed.
    RowsT(usize, usize, usize),
    /// Element-wise sum of two ranges.
    Sum(Range<usize>, Range<usize>),
}

// ---- reading ONNX ----

#[derive(Clone)]
struct OnnxTensor {
    dims: Vec<usize>,
    data_type: u64,
    data: Data,
}

/// Where a tensor's bytes live.
#[derive(Clone)]
enum Data {
    /// `raw_data` inside the .onnx file: path and byte range. Read lazily, like external data.
    Inline(PathBuf, Range<usize>),
    /// `float_data` / `int64_data`, decoded to little-endian bytes.
    Decoded(Arc<Vec<u8>>),
    /// An external data file: path, offset, length.
    External(PathBuf, u64, u64),
}

impl OnnxTensor {
    fn dtype(&self) -> std::result::Result<DType, String> {
        match self.data_type {
            1 => Ok(DType::F32),
            7 => Ok(DType::I64),
            t => Err(format!("unsupported ONNX tensor type {t}")),
        }
    }

    fn bytes(&self) -> std::result::Result<Bytes, String> {
        Ok(match &self.data {
            Data::Inline(file, r) => Bytes::from_file(file.clone(), r.len() as u64, r.start as u64),
            Data::Decoded(b) => Bytes::from_bytes_vec(b.to_vec()),
            Data::External(file, offset, len) => Bytes::from_file(file.clone(), *len, *offset),
        })
    }

    fn f32s(&self) -> std::result::Result<Vec<f32>, String> {
        let (file, offset, len) = match &self.data {
            Data::Inline(file, r) => (file, r.start as u64, r.len()),
            Data::External(file, offset, len) => (file, *offset, *len as usize),
            Data::Decoded(b) => return Ok(le_f32s(b)),
        };
        use std::io::{Read, Seek, SeekFrom};
        let io = |e: std::io::Error| format!("{}: {e}", file.display());
        let mut f = std::fs::File::open(file).map_err(io)?;
        f.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut raw = vec![0u8; len];
        f.read_exact(&mut raw).map_err(io)?;
        Ok(le_f32s(&raw))
    }
}

fn le_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Custom metadata (`metadata_props`), graph inputs and graph output names of an ONNX model.
pub(crate) struct ModelInfo {
    pub metadata: HashMap<String, String>,
    /// Input names and shapes; symbolic dimensions are -1.
    pub inputs: Vec<(String, Vec<i64>)>,
    /// Read by Sortformer only.
    #[cfg_attr(not(feature = "sortformer"), allow(dead_code))]
    pub outputs: Vec<String>,
}

pub(crate) fn model_info(path: &Path) -> Result<ModelInfo> {
    let bad = |what: &str| {
        Error::Model(format!(
            "{}: not a valid ONNX model ({what})",
            path.display()
        ))
    };
    let buf = std::fs::read(path)?;
    let text = |r: &Range<usize>| String::from_utf8_lossy(&buf[r.clone()]).into_owned();
    let model = fields(&buf, 0..buf.len()).map_err(|e| bad(&e))?;
    let mut metadata = HashMap::new();
    // ModelProto.metadata_props: key/value entries
    for entry in model.iter().filter(|f| f.0 == 14) {
        let kv = fields(&buf, entry.2.clone()).map_err(|e| bad(&e))?;
        let get = |n| kv.iter().find(|f| f.0 == n).map(|f| text(&f.2));
        if let (Some(k), Some(v)) = (get(1), get(2)) {
            metadata.insert(k, v);
        }
    }
    let graph = model
        .iter()
        .find(|f| f.0 == 7)
        .ok_or_else(|| bad("no graph"))?;
    let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
    let graph_fields = fields(&buf, graph.2.clone()).map_err(|e| bad(&e))?;
    // GraphProto.input / .output: ValueInfoProto { name: 1, type: 2 }
    for (num, _, value) in graph_fields.iter().filter(|f| f.0 == 11 || f.0 == 12) {
        let info = fields(&buf, value.clone()).map_err(|e| bad(&e))?;
        let Some(name) = info.iter().find(|f| f.0 == 1).map(|f| text(&f.2)) else {
            continue;
        };
        if *num == 12 {
            outputs.push(name);
            continue;
        }
        // TypeProto.tensor_type (1) -> TypeProto.Tensor.shape (2) -> dim (1): dim_value (1)
        let mut dims = Vec::new();
        let nested = |f: Option<&Field>, n: u64| -> std::result::Result<Option<Field>, String> {
            match f {
                Some(f) => Ok(fields(&buf, f.2.clone())?.into_iter().find(|g| g.0 == n)),
                None => Ok(None),
            }
        };
        let tensor_type = nested(info.iter().find(|f| f.0 == 2), 1).map_err(|e| bad(&e))?;
        let shape = nested(tensor_type.as_ref(), 2).map_err(|e| bad(&e))?;
        if let Some(shape) = shape {
            for dim in fields(&buf, shape.2)
                .map_err(|e| bad(&e))?
                .iter()
                .filter(|f| f.0 == 1)
            {
                let dim = fields(&buf, dim.2.clone()).map_err(|e| bad(&e))?;
                dims.push(
                    dim.iter()
                        .find(|f| f.0 == 1 && f.1 == 0)
                        .map_or(-1, |f| f.2.start as i64),
                );
            }
        }
        inputs.push((name, dims));
    }
    Ok(ModelInfo {
        metadata,
        inputs,
        outputs,
    })
}

/// Every initializer and `Constant` node tensor of an ONNX model, by [`weight_key`].
fn read_tensors(path: &Path) -> Result<HashMap<String, OnnxTensor>> {
    let bad = |what: &str| {
        Error::Model(format!(
            "{}: not a valid ONNX model ({what})",
            path.display()
        ))
    };
    let buf = std::fs::read(path)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let model = fields(&buf, 0..buf.len()).map_err(|e| bad(&e))?;
    let graph = model
        .iter()
        .find(|f| f.0 == 7)
        .ok_or_else(|| bad("no graph"))?;

    let mut named = HashMap::new();
    let mut consumers: HashMap<String, Vec<String>> = HashMap::new();
    for (num, _, val) in fields(&buf, graph.2.clone()).map_err(|e| bad(&e))? {
        match num {
            // GraphProto.initializer
            5 => {
                let (name, t) = tensor(&buf, val, path, dir).map_err(|e| bad(&e))?;
                named.insert(name, t);
            }
            // GraphProto.node
            1 => {
                let node = fields(&buf, val).map_err(|e| bad(&e))?;
                let text = |f: &Field| String::from_utf8_lossy(&buf[f.2.clone()]).into_owned();
                let name = node.iter().find(|f| f.0 == 3).map(text).unwrap_or_default();
                for (i, input) in node.iter().filter(|f| f.0 == 1).enumerate() {
                    consumers
                        .entry(text(input))
                        .or_default()
                        .push(format!("{name}#{i}"));
                }
                if !node
                    .iter()
                    .any(|f| f.0 == 4 && buf[f.2.clone()] == *b"Constant")
                {
                    continue;
                }
                let Some(output) = node.iter().find(|f| f.0 == 2).map(text) else {
                    continue;
                };
                for attr in node.iter().filter(|f| f.0 == 5) {
                    let attr = fields(&buf, attr.2.clone()).map_err(|e| bad(&e))?;
                    let is_value = attr
                        .iter()
                        .any(|f| f.0 == 1 && buf[f.2.clone()] == *b"value");
                    if let (true, Some(t)) = (is_value, attr.iter().find(|f| f.0 == 5)) {
                        let (_, t) = tensor(&buf, t.2.clone(), path, dir).map_err(|e| bad(&e))?;
                        named.insert(output.clone(), t);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(named
        .into_iter()
        .map(|(name, t)| (weight_key(&name, consumers.get(&name)), t))
        .collect())
}

/// A protobuf field: number, wire type, and the value's byte range (length-delimited and fixed
/// fields) or the varint value itself as `v..v`.
type Field = (u64, u8, Range<usize>);

fn varint(buf: &[u8], pos: &mut usize) -> std::result::Result<u64, String> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf.get(*pos).ok_or("truncated varint")?;
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err("varint too long".into())
}

fn fields(buf: &[u8], range: Range<usize>) -> std::result::Result<Vec<Field>, String> {
    let mut out = Vec::new();
    let mut pos = range.start;
    while pos < range.end {
        let key = varint(buf, &mut pos)?;
        let (num, wire) = (key >> 3, (key & 7) as u8);
        let len = match wire {
            0 => {
                let v = varint(buf, &mut pos)? as usize;
                out.push((num, wire, v..v));
                continue;
            }
            1 => 8,
            2 => varint(buf, &mut pos)? as usize,
            5 => 4,
            w => return Err(format!("unsupported protobuf wire type {w}")),
        };
        let end = pos
            .checked_add(len)
            .filter(|&e| e <= range.end)
            .ok_or("field runs past its message")?;
        out.push((num, wire, pos..end));
        pos = end;
    }
    Ok(out)
}

/// Parse a TensorProto: its name, and where its data lives.
fn tensor(
    buf: &[u8],
    range: Range<usize>,
    file: &Path,
    dir: &Path,
) -> std::result::Result<(String, OnnxTensor), String> {
    let (mut name, mut dims, mut data_type) = (String::new(), Vec::new(), 0u64);
    let (mut raw, mut decoded) = (None, Vec::<u8>::new());
    let mut external: HashMap<String, String> = HashMap::new();
    for (num, wire, val) in fields(buf, range)? {
        match (num, wire) {
            (1, 0) => dims.push(val.start),
            (1, 2) => {
                let mut p = val.start;
                while p < val.end {
                    dims.push(varint(buf, &mut p)? as usize);
                }
            }
            (2, 0) => data_type = val.start as u64,
            // float_data: packed or not, already little-endian f32
            (4, 2) | (4, 5) => decoded.extend_from_slice(&buf[val]),
            // int64_data: varints
            (7, 2) => {
                let mut p = val.start;
                while p < val.end {
                    decoded.extend_from_slice(&(varint(buf, &mut p)? as i64).to_le_bytes());
                }
            }
            (7, 0) => decoded.extend_from_slice(&(val.start as i64).to_le_bytes()),
            (8, 2) => name = String::from_utf8_lossy(&buf[val]).into_owned(),
            (9, 2) => raw = Some(val),
            // external_data: key/value entries
            (13, 2) => {
                let kv = fields(buf, val)?;
                let get = |n| {
                    kv.iter()
                        .find(|f| f.0 == n)
                        .map(|f| String::from_utf8_lossy(&buf[f.2.clone()]).into_owned())
                };
                if let (Some(k), Some(v)) = (get(1), get(2)) {
                    external.insert(k, v);
                }
            }
            _ => {}
        }
    }
    let elem = match data_type {
        1 => 4,
        7 => 8,
        _ => 0,
    };
    let data = if let Some(location) = external.get("location") {
        let number = |key: &str| {
            external
                .get(key)
                .map(|v| v.parse::<u64>().map_err(|e| format!("{name}: {key}: {e}")))
                .transpose()
        };
        let offset = number("offset")?.unwrap_or(0);
        let length = number("length")?.unwrap_or((dims.iter().product::<usize>() * elem) as u64);
        Data::External(dir.join(location), offset, length)
    } else if let Some(r) = raw {
        Data::Inline(file.to_path_buf(), r)
    } else {
        Data::Decoded(Arc::new(decoded))
    };
    Ok((
        name,
        OnnxTensor {
            dims,
            data_type,
            data,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a tiny protobuf writer, enough to build ONNX test models ----

    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    fn bytes_field(num: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(num << 3 | 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    fn int_field(num: u64, v: u64) -> Vec<u8> {
        let mut out = varint(num << 3);
        out.extend(varint(v));
        out
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// TensorProto with float data, inline or (with `external`) in a data file.
    fn tensor_proto(
        name: &str,
        dims: &[u64],
        values: &[f32],
        external: Option<(&str, u64)>,
    ) -> Vec<u8> {
        let mut t: Vec<u8> = dims.iter().flat_map(|&d| int_field(1, d)).collect();
        t.extend(int_field(2, 1));
        t.extend(bytes_field(8, name.as_bytes()));
        match external {
            None => t.extend(bytes_field(9, &f32_bytes(values))),
            Some((file, offset)) => {
                for (k, v) in [
                    ("location", file.to_string()),
                    ("offset", offset.to_string()),
                    ("length", (values.len() * 4).to_string()),
                ] {
                    t.extend(bytes_field(
                        13,
                        &[bytes_field(1, k.as_bytes()), bytes_field(2, v.as_bytes())].concat(),
                    ));
                }
                t.extend(int_field(14, 1));
            }
        }
        t
    }

    fn node(
        name: &str,
        op: &str,
        inputs: &[&str],
        outputs: &[&str],
        value: Option<Vec<u8>>,
    ) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            n.extend(bytes_field(1, i.as_bytes()));
        }
        for o in outputs {
            n.extend(bytes_field(2, o.as_bytes()));
        }
        n.extend(bytes_field(3, name.as_bytes()));
        n.extend(bytes_field(4, op.as_bytes()));
        if let Some(t) = value {
            n.extend(bytes_field(
                5,
                &[
                    bytes_field(1, b"value"),
                    bytes_field(5, &t),
                    int_field(20, 4),
                ]
                .concat(),
            ));
        }
        n
    }

    const LSTM_W: [f32; 16] = [
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ];

    /// An ONNX model holding one of each kind of tensor the loader handles, plus its external
    /// data file, in a fresh temporary directory.
    fn test_model() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "parakeet-rs-onnx-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // external data: 4 padding floats, then the LSTM weight
        let data = f32_bytes(&[[9.0; 4].as_slice(), &LSTM_W].concat());
        std::fs::write(dir.join("w.data"), data).unwrap();

        let table: Vec<f32> = (0..10).map(|v| v as f32).collect();
        let mut graph = Vec::new();
        graph.extend(bytes_field(
            1,
            &node(
                "/a/b/MatMul",
                "MatMul",
                &["x", "onnx::MatMul_9"],
                &["y"],
                None,
            ),
        ));
        graph.extend(bytes_field(
            1,
            &node(
                "/layers.3/conv/MatMul",
                "MatMul",
                &["y", "onnx::MatMul_10"],
                &["z"],
                None,
            ),
        ));
        graph.extend(bytes_field(
            1,
            &node(
                "/lstm/LSTM",
                "LSTM",
                &["z", "onnx::LSTM_1", "onnx::LSTM_1", "onnx::LSTM_2"],
                &["h"],
                None,
            ),
        ));
        graph.extend(bytes_field(
            1,
            &node(
                "/pos_enc/Constant",
                "Constant",
                &[],
                &["onnx::Slice_5"],
                Some(tensor_proto("", &[1, 5, 2], &table, None)),
            ),
        ));
        graph.extend(bytes_field(
            1,
            &node("/pos_enc/Slice", "Slice", &["onnx::Slice_5"], &["p"], None),
        ));
        graph.extend(bytes_field(
            5,
            &tensor_proto("layers.0.norm.weight", &[2], &[1.0, 2.0], None),
        ));
        graph.extend(bytes_field(
            5,
            &tensor_proto("onnx::MatMul_9", &[2, 3], &[1., 2., 3., 4., 5., 6.], None),
        ));
        graph.extend(bytes_field(
            5,
            &tensor_proto("onnx::MatMul_10", &[2, 3], &[1., 2., 3., 4., 5., 6.], None),
        ));
        graph.extend(bytes_field(
            5,
            &tensor_proto("onnx::LSTM_1", &[1, 8, 2], &LSTM_W, Some(("w.data", 16))),
        ));
        let bias: Vec<f32> = (0..16).map(|v| v as f32 * 10.0).collect();
        graph.extend(bytes_field(
            5,
            &tensor_proto("onnx::LSTM_2", &[1, 16], &bias, None),
        ));
        let model = [int_field(1, 8), bytes_field(7, &graph)].concat();
        let path = dir.join("model.onnx");
        std::fs::write(&path, model).unwrap();
        path
    }

    fn values(t: PackTensor) -> Vec<f32> {
        t.to_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn load_one(
        tensors: &HashMap<String, OnnxTensor>,
        shape: Option<&'static [usize]>,
        key: &'static str,
        transform: Transform,
    ) -> Vec<f32> {
        let w = Weight {
            path: "p",
            shape,
            source: Source::Param(key, transform),
        };
        let (src, layout) = resolve(tensors, key).expect("key resolves");
        values(param(&w, src, layout, transform).unwrap())
    }

    #[test]
    fn keys_do_not_depend_on_anonymous_names() {
        let tensors = read_tensors(&test_model()).unwrap();
        let mut keys: Vec<_> = tensors.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "/a/b/MatMul#1",
                "/layers.3/conv/MatMul#1",
                "/lstm/LSTM#1|/lstm/LSTM#2",
                "/lstm/LSTM#3",
                "/pos_enc/Slice#0",
                "layers.0.norm.weight"
            ]
        );
    }

    #[test]
    fn copies_inline_and_external_tensors() {
        let tensors = read_tensors(&test_model()).unwrap();
        assert_eq!(
            load_one(
                &tensors,
                Some(&[2]),
                "layers.0.norm.weight",
                Transform::Copy
            ),
            [1.0, 2.0]
        );
        assert_eq!(
            load_one(&tensors, None, "/lstm/LSTM#1|/lstm/LSTM#2", Transform::Copy),
            LSTM_W
        );
    }

    #[test]
    fn lstm_gates_are_split_and_biases_folded() {
        let tensors = read_tensors(&test_model()).unwrap();
        // gate rows 2..4 of the packed [1, 8, 2] weight, transposed to [2, 2]
        assert_eq!(
            load_one(
                &tensors,
                Some(&[2, 2]),
                "/lstm/LSTM#1|/lstm/LSTM#2",
                Transform::RowsT(2, 4)
            ),
            [4., 6., 5., 7.]
        );
        // input bias of gate 0 = W bias 0..2 + R bias 8..10
        assert_eq!(
            load_one(
                &tensors,
                Some(&[2]),
                "/lstm/LSTM#3",
                Transform::BiasSum(0, 2, 8, 10)
            ),
            [80., 100.]
        );
    }

    #[test]
    fn longer_positional_tables_are_center_sliced() {
        let tensors = read_tensors(&test_model()).unwrap();
        // [1, 5, 2] -> the middle 3 rows
        assert_eq!(
            load_one(
                &tensors,
                Some(&[1, 3, 2]),
                "/pos_enc/Slice#0",
                Transform::CenterRows
            ),
            [2., 3., 4., 5., 6., 7.]
        );
    }

    #[test]
    fn other_export_layouts_are_transposed() {
        let tensors = read_tensors(&test_model()).unwrap();
        // pointwise conv exported as MatMul [in, out] -> reference Conv1d [out, in, 1]
        assert_eq!(
            load_one(
                &tensors,
                Some(&[3, 2, 1]),
                "layers.3.conv.pointwise_conv1.weight",
                Transform::Copy
            ),
            [1., 4., 2., 5., 3., 6.]
        );
    }

    #[test]
    fn missing_and_mismatched_weights_are_errors() {
        let tensors = read_tensors(&test_model()).unwrap();
        assert!(resolve(&tensors, "layers.9.norm.weight").is_none());
        let w = Weight {
            path: "p",
            shape: Some(&[3]),
            source: Source::Param("layers.0.norm.weight", Transform::Copy),
        };
        let (src, layout) = resolve(&tensors, "layers.0.norm.weight").unwrap();
        assert!(param(&w, src, layout, Transform::Copy).is_err());
        assert!(
            read_tensors(&std::env::temp_dir().join("parakeet-rs-no-such-model.onnx")).is_err()
        );
    }

    #[test]
    fn truncated_protobuf_is_an_error_not_a_panic() {
        let path = test_model();
        let bytes = std::fs::read(&path).unwrap();
        let cut = path.with_file_name("cut.onnx");
        std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
        assert!(read_tensors(&cut).is_err());
    }
}
