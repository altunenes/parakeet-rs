use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::path::Path;

/// Canary 1B model - multilingual ASR with AED architecture
pub struct CanaryModel {
    pub encoder: Session,
    pub decoder: Session,
}

impl CanaryModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_path = Self::find_decoder(model_dir)?;

        let session_builder = Session::builder()
            .map_err(|e| Error::Model(format!("Failed to create session builder: {e}")))?;

        let session_builder = exec_config.apply_to_session_builder(session_builder)?;

        let encoder = session_builder
            .clone()
            .commit_from_file(&encoder_path)
            .map_err(|e| Error::Model(format!("Failed to load encoder: {e}")))?;

        let decoder = session_builder
            .commit_from_file(&decoder_path)
            .map_err(|e| Error::Model(format!("Failed to load decoder: {e}")))?;

        Ok(Self { encoder, decoder })
    }

    fn find_encoder(dir: &Path) -> Result<std::path::PathBuf> {
        let candidates = [
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ];

        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        // Fallback: find any encoder ONNX file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("encoder") && name.ends_with(".onnx") {
                        return Ok(path);
                    }
                }
            }
        }

        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder(dir: &Path) -> Result<std::path::PathBuf> {
        let candidates = [
            "decoder-model.onnx",
            "decoder.onnx",
            "decoder-model.int8.onnx",
        ];

        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        // Fallback: find any decoder ONNX file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("decoder") && name.ends_with(".onnx") {
                        return Ok(path);
                    }
                }
            }
        }

        Err(Error::Config(format!(
            "No decoder model found in {}",
            dir.display()
        )))
    }

    /// Encode audio features to embeddings
    pub fn encode(
        &mut self,
        audio_signal: Array3<f32>,
        length: Array1<i64>,
    ) -> Result<(Array3<f32>, Array2<i64>)> {
        let input_value = ort::value::Value::from_array(audio_signal)?;
        let length_value = ort::value::Value::from_array(length)?;

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => input_value,
            "length" => length_value
        ))?;

        let encoder_embeddings_value = &outputs["encoder_embeddings"];
        let encoder_mask_value = &outputs["encoder_mask"];

        let (encoder_embeddings_shape, encoder_embeddings_data) = encoder_embeddings_value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder embeddings: {e}")))?;

        let (encoder_mask_shape, encoder_mask_data) = encoder_mask_value
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder mask: {e}")))?;

        let encoder_embeddings = Array3::from_shape_vec(
            (
                encoder_embeddings_shape[0] as usize,
                encoder_embeddings_shape[1] as usize,
                encoder_embeddings_shape[2] as usize,
            ),
            encoder_embeddings_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape encoder embeddings: {e}")))?;

        let encoder_mask = Array2::from_shape_vec(
            (
                encoder_mask_shape[0] as usize,
                encoder_mask_shape[1] as usize,
            ),
            encoder_mask_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape encoder mask: {e}")))?;

        Ok((encoder_embeddings, encoder_mask))
    }

    /// Decode with autoregressive attention decoder
    pub fn decode(
        &mut self,
        input_ids: Array2<i64>,
        encoder_embeddings: Array3<f32>,
        encoder_mask: Array2<i64>,
        decoder_mems: Array4<f32>,
    ) -> Result<(Array3<f32>, Array4<f32>)> {
        let input_ids_value = ort::value::Value::from_array(input_ids)?;
        let encoder_embeddings_value = ort::value::Value::from_array(encoder_embeddings)?;
        let encoder_mask_value = ort::value::Value::from_array(encoder_mask)?;
        let decoder_mems_value = ort::value::Value::from_array(decoder_mems)?;

        let outputs = self.decoder.run(ort::inputs!(
            "input_ids" => input_ids_value,
            "encoder_embeddings" => encoder_embeddings_value,
            "encoder_mask" => encoder_mask_value,
            "decoder_mems" => decoder_mems_value
        ))?;

        let logits_value = &outputs["logits"];
        let decoder_hidden_states_value = &outputs["decoder_hidden_states"];

        let (logits_shape, logits_data) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        let (hidden_shape, hidden_data) =
            decoder_hidden_states_value
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract hidden states: {e}")))?;

        let logits = Array3::from_shape_vec(
            (
                logits_shape[0] as usize,
                logits_shape[1] as usize,
                logits_shape[2] as usize,
            ),
            logits_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape logits: {e}")))?;

        let decoder_hidden_states = Array4::from_shape_vec(
            (
                hidden_shape[0] as usize,
                hidden_shape[1] as usize,
                hidden_shape[2] as usize,
                hidden_shape[3] as usize,
            ),
            hidden_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape hidden states: {e}")))?;

        Ok((logits, decoder_hidden_states))
    }
}
