use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array0, Array2, Array4, Array5};
use ort::session::Session;
use std::path::Path;

/// Cohere Transcribe model architecture constants.
/// Derived from ONNX graph inspection of the Handy INT8 export.
const NUM_DECODER_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const MAX_SEQ_LEN: usize = 1024;

/// ONNX session wrapper for Cohere Transcribe encoder + decoder.
pub(crate) struct CohereModel {
    encoder: Session,
    decoder: Session,
}

/// Cross-attention key/value pairs from the encoder.
/// Shape: [num_layers=8, batch=1, T_enc, hidden_dim=1024]
pub(crate) struct CohereEncoderOutput {
    pub(crate) cross_k: Array4<f32>,
    pub(crate) cross_v: Array4<f32>,
}

/// Self-attention KV cache for the decoder.
/// Shape: [num_layers=8, batch=1, num_heads=8, max_seq=1024, head_dim=128]
pub(crate) struct CohereSelfCache {
    pub(crate) k: Array5<f32>,
    pub(crate) v: Array5<f32>,
}

impl CohereSelfCache {
    pub(crate) fn empty() -> Self {
        Self {
            k: Array5::zeros((NUM_DECODER_LAYERS, 1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)),
            v: Array5::zeros((NUM_DECODER_LAYERS, 1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)),
        }
    }
}

impl CohereModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = Self::find_file(model_dir, &[
            "cohere-encoder.int8.onnx",
            "cohere-encoder.onnx",
            "encoder.int8.onnx",
            "encoder.onnx",
        ])?;
        let decoder_path = Self::find_file(model_dir, &[
            "cohere-decoder.int8.onnx",
            "cohere-decoder.onnx",
            "decoder.int8.onnx",
            "decoder.onnx",
        ])?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let decoder = builder.commit_from_file(&decoder_path)?;

        Ok(Self { encoder, decoder })
    }

    /// Run the encoder on raw 16kHz mono f32 audio samples.
    pub(crate) fn run_encoder(&mut self, audio: &[f32]) -> Result<CohereEncoderOutput> {
        let audio_array = Array2::from_shape_vec((1, audio.len()), audio.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create audio tensor: {e}")))?;

        let outputs = self.encoder.run(ort::inputs![
            "audio" => ort::value::Value::from_array(audio_array)?
        ])?;

        let (k_shape, k_data) = outputs["n_layer_cross_k"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cross_k: {e}")))?;

        let (v_shape, v_data) = outputs["n_layer_cross_v"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract cross_v: {e}")))?;

        let cross_k = Array4::from_shape_vec(
            (
                k_shape[0] as usize,
                k_shape[1] as usize,
                k_shape[2] as usize,
                k_shape[3] as usize,
            ),
            k_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape cross_k: {e}")))?;

        let cross_v = Array4::from_shape_vec(
            (
                v_shape[0] as usize,
                v_shape[1] as usize,
                v_shape[2] as usize,
                v_shape[3] as usize,
            ),
            v_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape cross_v: {e}")))?;

        Ok(CohereEncoderOutput { cross_k, cross_v })
    }

    /// Run a single decoder step.
    pub(crate) fn run_decoder_step(
        &mut self,
        tokens: &Array2<i64>,
        cache: &CohereSelfCache,
        encoder_out: &CohereEncoderOutput,
        offset: i64,
    ) -> Result<(Array2<f32>, CohereSelfCache)> {
        let offset_arr = Array0::from_elem((), offset);

        // Use borrowed tensor views to avoid cloning large arrays each step.
        // The encoder cross-attention outputs (~4MB each) and KV cache (~32MB each)
        // are read-only inputs; only the output cache is a new allocation.
        let tokens_ref = ort::value::TensorRef::<i64>::from_array_view(tokens.view())?;
        let cache_k_ref = ort::value::TensorRef::<f32>::from_array_view(cache.k.view())?;
        let cache_v_ref = ort::value::TensorRef::<f32>::from_array_view(cache.v.view())?;
        let cross_k_ref = ort::value::TensorRef::<f32>::from_array_view(encoder_out.cross_k.view())?;
        let cross_v_ref = ort::value::TensorRef::<f32>::from_array_view(encoder_out.cross_v.view())?;

        let outputs = self.decoder.run(ort::inputs!(
            "tokens" => tokens_ref,
            "in_n_layer_self_k_cache" => cache_k_ref,
            "in_n_layer_self_v_cache" => cache_v_ref,
            "n_layer_cross_k" => cross_k_ref,
            "n_layer_cross_v" => cross_v_ref,
            "offset" => ort::value::Value::from_array(offset_arr)?
        ))?;

        // Logits: [batch, n_tokens, vocab_size] - extract last token's logits
        let (l_shape, l_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        let n_tokens = l_shape[1] as usize;
        let vocab_size = l_shape[2] as usize;
        let logits = Array2::from_shape_vec((n_tokens, vocab_size), l_data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to reshape logits: {e}")))?;

        // Updated KV cache
        let (ok_shape, ok_data) = outputs["out_n_layer_self_k_cache"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract k_cache: {e}")))?;

        let (ov_shape, ov_data) = outputs["out_n_layer_self_v_cache"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract v_cache: {e}")))?;

        let new_k = Array5::from_shape_vec(
            (
                ok_shape[0] as usize,
                ok_shape[1] as usize,
                ok_shape[2] as usize,
                ok_shape[3] as usize,
                ok_shape[4] as usize,
            ),
            ok_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape k_cache: {e}")))?;

        let new_v = Array5::from_shape_vec(
            (
                ov_shape[0] as usize,
                ov_shape[1] as usize,
                ov_shape[2] as usize,
                ov_shape[3] as usize,
                ov_shape[4] as usize,
            ),
            ov_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape v_cache: {e}")))?;

        Ok((logits, CohereSelfCache { k: new_k, v: new_v }))
    }

    fn find_file(dir: &Path, candidates: &[&str]) -> Result<std::path::PathBuf> {
        for name in candidates {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "None of {:?} found in {}",
            candidates, dir.display()
        )))
    }
}
