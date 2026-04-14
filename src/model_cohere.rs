use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::path::{Path, PathBuf};

/// Cohere Transcribe architecture constants.
/// Verified against `CohereLabs/cohere-transcribe-03-2026` config and the
/// onnx-community ONNX export I/O signatures.
pub(crate) const NUM_DECODER_LAYERS: usize = 8;
pub(crate) const NUM_KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const N_MELS: usize = 128;

/// Encoder hidden states.
/// Shape: `[batch=1, T_enc, HIDDEN_SIZE]`
pub(crate) struct CohereEncoderOutput {
    pub(crate) hidden_states: Array3<f32>,
}

/// Per-layer cache tensor for one of `{decoder, encoder}.{key, value}`.
/// Shape: `[batch=1, NUM_KV_HEADS, seq_len, HEAD_DIM]`.
type LayerCache = Array4<f32>;

/// Decoder `past_key_values` cache split into self-attention (`decoder_*`)
/// and cross-attention (`encoder_*`) tensors per layer.
///
/// On the first decoder call all caches are zero-length; the model
/// populates the cross-attention caches from `encoder_hidden_states`. On
/// subsequent calls the model writes new self-attention K/V into the
/// growing decoder caches and reuses the encoder caches as-is.
pub(crate) struct CoherePastKv {
    pub(crate) decoder_k: Vec<LayerCache>,
    pub(crate) decoder_v: Vec<LayerCache>,
    pub(crate) encoder_k: Vec<LayerCache>,
    pub(crate) encoder_v: Vec<LayerCache>,
}

impl CoherePastKv {
    pub(crate) fn empty() -> Self {
        let zero = Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));
        Self {
            decoder_k: vec![zero.clone(); NUM_DECODER_LAYERS],
            decoder_v: vec![zero.clone(); NUM_DECODER_LAYERS],
            encoder_k: vec![zero.clone(); NUM_DECODER_LAYERS],
            encoder_v: vec![zero; NUM_DECODER_LAYERS],
        }
    }

    /// Total decoded tokens accumulated in the self-attention cache.
    pub(crate) fn past_decoder_len(&self) -> usize {
        self.decoder_k[0].shape()[2]
    }
}

pub(crate) struct CohereModel {
    encoder: Session,
    decoder: Session,
}

impl CohereModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Try int8 quantised first, then fp32, then fp16. Both flat
        // (`encoder_model_quantized.onnx` next to the directory root) and
        // nested (`onnx/encoder_model_quantized.onnx` as in the
        // onnx-community HF repo) layouts are supported.
        let encoder_path = Self::find_file(
            model_dir,
            &[
                "onnx/encoder_model_quantized.onnx",
                "encoder_model_quantized.onnx",
                "onnx/encoder_model.onnx",
                "encoder_model.onnx",
                "onnx/encoder_model_fp16.onnx",
                "encoder_model_fp16.onnx",
            ],
        )?;
        let decoder_path = Self::find_file(
            model_dir,
            &[
                "onnx/decoder_model_merged_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
                "onnx/decoder_model_merged_fp16.onnx",
                "decoder_model_merged_fp16.onnx",
            ],
        )?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let decoder = builder.commit_from_file(&decoder_path)?;

        Ok(Self { encoder, decoder })
    }

    /// Run the encoder on log-mel features with shape `[1, T, N_MELS]`.
    pub(crate) fn run_encoder(
        &mut self,
        input_features: &Array3<f32>,
    ) -> Result<CohereEncoderOutput> {
        let features_ref = ort::value::TensorRef::<f32>::from_array_view(input_features.view())?;
        let outputs = self.encoder.run(ort::inputs!(
            "input_features" => features_ref,
        ))?;

        let (h_shape, h_data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract last_hidden_state: {e}")))?;

        let hidden_states = Array3::from_shape_vec(
            (
                h_shape[0] as usize,
                h_shape[1] as usize,
                h_shape[2] as usize,
            ),
            h_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape last_hidden_state: {e}")))?;

        Ok(CohereEncoderOutput { hidden_states })
    }

    /// Run one decoder step. Returns logits for the LAST output position
    /// only (we always pass `num_logits_to_keep=1` since we only need the
    /// next-token distribution for greedy decoding) and the new past_kv.
    pub(crate) fn run_decoder_step(
        &mut self,
        tokens: &Array2<i64>,
        past_kv: &CoherePastKv,
        encoder_out: &CohereEncoderOutput,
    ) -> Result<(Array1<f32>, CoherePastKv)> {
        let seq_len = tokens.shape()[1];
        let past_len = past_kv.past_decoder_len();
        let total_len = past_len + seq_len;

        // position_ids: [past_len, past_len+1, ..., past_len+seq_len-1]
        let position_ids = Array2::from_shape_vec(
            (1, seq_len),
            (past_len..total_len).map(|i| i as i64).collect(),
        )
        .map_err(|e| Error::Model(format!("position_ids shape mismatch: {e}")))?;

        // attention_mask: [1, total_len] all-ones (covers past + current tokens)
        let attention_mask = Array2::<i64>::from_elem((1, total_len), 1);

        // num_logits_to_keep: scalar i64 = 1 (only need the last position)
        let num_logits = ndarray::Array0::<i64>::from_elem((), 1);

        // Borrow large tensors as views to avoid per-step copies.
        let tokens_ref = ort::value::TensorRef::<i64>::from_array_view(tokens.view())?;
        let attn_ref = ort::value::TensorRef::<i64>::from_array_view(attention_mask.view())?;
        let pos_ref = ort::value::TensorRef::<i64>::from_array_view(position_ids.view())?;
        let enc_ref =
            ort::value::TensorRef::<f32>::from_array_view(encoder_out.hidden_states.view())?;

        let dk0 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[0].view())?;
        let dv0 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[0].view())?;
        let ek0 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[0].view())?;
        let ev0 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[0].view())?;
        let dk1 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[1].view())?;
        let dv1 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[1].view())?;
        let ek1 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[1].view())?;
        let ev1 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[1].view())?;
        let dk2 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[2].view())?;
        let dv2 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[2].view())?;
        let ek2 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[2].view())?;
        let ev2 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[2].view())?;
        let dk3 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[3].view())?;
        let dv3 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[3].view())?;
        let ek3 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[3].view())?;
        let ev3 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[3].view())?;
        let dk4 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[4].view())?;
        let dv4 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[4].view())?;
        let ek4 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[4].view())?;
        let ev4 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[4].view())?;
        let dk5 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[5].view())?;
        let dv5 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[5].view())?;
        let ek5 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[5].view())?;
        let ev5 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[5].view())?;
        let dk6 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[6].view())?;
        let dv6 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[6].view())?;
        let ek6 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[6].view())?;
        let ev6 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[6].view())?;
        let dk7 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_k[7].view())?;
        let dv7 = ort::value::TensorRef::<f32>::from_array_view(past_kv.decoder_v[7].view())?;
        let ek7 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_k[7].view())?;
        let ev7 = ort::value::TensorRef::<f32>::from_array_view(past_kv.encoder_v[7].view())?;

        let outputs = self.decoder.run(ort::inputs!(
            "input_ids" => tokens_ref,
            "attention_mask" => attn_ref,
            "position_ids" => pos_ref,
            "num_logits_to_keep" => ort::value::Value::from_array(num_logits)?,
            "encoder_hidden_states" => enc_ref,
            "past_key_values.0.decoder.key" => dk0,
            "past_key_values.0.decoder.value" => dv0,
            "past_key_values.0.encoder.key" => ek0,
            "past_key_values.0.encoder.value" => ev0,
            "past_key_values.1.decoder.key" => dk1,
            "past_key_values.1.decoder.value" => dv1,
            "past_key_values.1.encoder.key" => ek1,
            "past_key_values.1.encoder.value" => ev1,
            "past_key_values.2.decoder.key" => dk2,
            "past_key_values.2.decoder.value" => dv2,
            "past_key_values.2.encoder.key" => ek2,
            "past_key_values.2.encoder.value" => ev2,
            "past_key_values.3.decoder.key" => dk3,
            "past_key_values.3.decoder.value" => dv3,
            "past_key_values.3.encoder.key" => ek3,
            "past_key_values.3.encoder.value" => ev3,
            "past_key_values.4.decoder.key" => dk4,
            "past_key_values.4.decoder.value" => dv4,
            "past_key_values.4.encoder.key" => ek4,
            "past_key_values.4.encoder.value" => ev4,
            "past_key_values.5.decoder.key" => dk5,
            "past_key_values.5.decoder.value" => dv5,
            "past_key_values.5.encoder.key" => ek5,
            "past_key_values.5.encoder.value" => ev5,
            "past_key_values.6.decoder.key" => dk6,
            "past_key_values.6.decoder.value" => dv6,
            "past_key_values.6.encoder.key" => ek6,
            "past_key_values.6.encoder.value" => ev6,
            "past_key_values.7.decoder.key" => dk7,
            "past_key_values.7.decoder.value" => dv7,
            "past_key_values.7.encoder.key" => ek7,
            "past_key_values.7.encoder.value" => ev7,
        ))?;

        // logits: [1, n_positions, vocab_size]. With num_logits_to_keep=1
        // n_positions is 1, but we slice the last position defensively in
        // case the model ever returns all positions.
        let (l_shape, l_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
        let n_positions = l_shape[1] as usize;
        let vocab_size = l_shape[2] as usize;
        let last_start = n_positions.saturating_sub(1) * vocab_size;
        let logits = Array1::from_vec(l_data[last_start..last_start + vocab_size].to_vec());

        // Read all 32 present.* tensors into a new CoherePastKv
        let new_past = read_past_kv(&outputs)?;

        Ok((logits, new_past))
    }

    fn find_file(dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
        for name in candidates {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "None of {:?} found in {}",
            candidates,
            dir.display()
        )))
    }
}

/// Extract a 4-D `[1, NUM_KV_HEADS, seq, HEAD_DIM]` cache tensor from the
/// decoder outputs by name.
fn extract_cache(outputs: &ort::session::SessionOutputs, name: &str) -> Result<LayerCache> {
    let (shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract {name}: {e}")))?;
    Array4::from_shape_vec(
        (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        ),
        data.to_vec(),
    )
    .map_err(|e| Error::Model(format!("Failed to reshape {name}: {e}")))
}

fn read_past_kv(outputs: &ort::session::SessionOutputs) -> Result<CoherePastKv> {
    let mut decoder_k = Vec::with_capacity(NUM_DECODER_LAYERS);
    let mut decoder_v = Vec::with_capacity(NUM_DECODER_LAYERS);
    let mut encoder_k = Vec::with_capacity(NUM_DECODER_LAYERS);
    let mut encoder_v = Vec::with_capacity(NUM_DECODER_LAYERS);
    for i in 0..NUM_DECODER_LAYERS {
        decoder_k.push(extract_cache(outputs, &format!("present.{i}.decoder.key"))?);
        decoder_v.push(extract_cache(
            outputs,
            &format!("present.{i}.decoder.value"),
        )?);
        encoder_k.push(extract_cache(outputs, &format!("present.{i}.encoder.key"))?);
        encoder_v.push(extract_cache(
            outputs,
            &format!("present.{i}.encoder.value"),
        )?);
    }
    Ok(CoherePastKv {
        decoder_k,
        decoder_v,
        encoder_k,
        encoder_v,
    })
}
