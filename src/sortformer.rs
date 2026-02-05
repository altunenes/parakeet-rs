//! NVIDIA Sortformer v2 Streaming Speaker Diarization
//!
//! This module implements NVIDIA's Sortformer v2 streaming model for speaker diarization.
//!
//! Key features:
//! - Streaming inference with ~10s chunks (124 frames at 80ms each)
//! - FIFO buffer for context management
//! - Smart speaker cache compression (keeps important frames, not just recent)
//! - Silence profile tracking
//! - Post-processing: median filtering, hysteresis thresholding
//! - Supports up to 4 speakers
//!
//! Reference: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
//! Note that, my ONNX export:
//! CHUNK_LEN = 124
//! FIFO_LEN = 124
//! CACHE_LEN = 188
//! FEAT_DIM = 128
//! EMB_DIM = 512
//! Note, my stft code is adapted from: https://librosa.org/doc/main/generated/librosa.stft.html

use crate::error::{Error, Result};
use crate::execution::ModelConfig;
use ndarray::{s, Array1, Array2, Array3, Axis};
use ort::session::Session;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::path::Path;

// Model constants
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960464478e-8;
const SAMPLE_RATE: usize = 16000;

// Streaming constants
const CHUNK_LEN: usize = 124; // Frames per chunk (~10s at 80ms)
const FIFO_LEN: usize = 124; // FIFO buffer length
const SPKCACHE_LEN: usize = 188; // Speaker cache length
const SPKCACHE_UPDATE_PERIOD: usize = 124;
const SUBSAMPLING: usize = 8; // Audio frames -> model frames
const EMB_DIM: usize = 512; // Embedding dimension
const NUM_SPEAKERS: usize = 4; // Model supports 4 speakers
const FRAME_DURATION: f32 = 0.08; // 80ms per frame

// Cache compression params (from NeMo)
const SPKCACHE_SIL_FRAMES_PER_SPK: usize = 3;
const PRED_SCORE_THRESHOLD: f32 = 0.25;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;
const SIL_THRESHOLD: f32 = 0.2;
const MAX_INDEX: usize = 99999;

/// Post-processing configuration for speaker diarization. (NVIDIA official configs from v2 YAMLs)
///
/// Controls how raw model predictions are converted into speaker segments.
/// NVIDIA provides pre-tuned configs for different datasets (CallHome, DIHARD3, AMI).
///
/// # Parameters
/// - `onset`: Probability threshold to START a speaker segment (higher = more strict)
/// - `offset`: Probability threshold to END a speaker segment (lower = longer segments)
/// - `pad_onset`: Seconds to subtract from segment start times
/// - `pad_offset`: Seconds to add to segment end times
/// - `min_duration_on`: Minimum segment length in seconds (filters short blips)
/// - `min_duration_off`: Minimum gap between segments before merging
/// - `median_window`: Smoothing window size (odd number, higher = smoother)
///
/// # Pre-tuned Configs
/// - `callhome()` - (default)
/// - `dihard3()`
///
/// # Custom Config
/// Use `custom(onset, offset)` to create your own config for fine-tuning.
///
/// See: https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/speaker_tasks/diarization/conf/neural_diarizer
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    pub onset: f32,
    pub offset: f32,
    pub pad_onset: f32,
    pub pad_offset: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
    pub median_window: usize,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self::callhome()
    }
}

impl DiarizationConfig {
    /// CallHome dataset config for v2 (default)
    /// From: diar_streaming_sortformer_4spk-v2_callhome-part1.yaml
    pub fn callhome() -> Self {
        Self {
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            median_window: 11,
        }
    }

    /// DIHARD3 dataset config for v2
    /// From: diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml
    pub fn dihard3() -> Self {
        Self {
            onset: 0.56,
            offset: 1.0,
            pad_onset: 0.063,
            pad_offset: 0.002,
            min_duration_on: 0.007,
            min_duration_off: 0.151,
            median_window: 11,
        }
    }

    /// Create a custom config for fine-tuning diarization behavior.
    ///
    /// # Arguments
    /// * `onset` - Probability threshold to start a segment (0.0-1.0, typical: 0.5-0.7)
    /// * `offset` - Probability threshold to end a segment (0.0-1.0, typical: 0.4-0.6)
    ///
    /// # Example
    /// ```rust
    /// use parakeet_rs::sortformer::DiarizationConfig;
    ///
    /// // More sensitive detection (lower thresholds)
    /// let sensitive = DiarizationConfig::custom(0.5, 0.4);
    ///
    /// // Stricter detection (higher thresholds, fewer false positives)
    /// let strict = DiarizationConfig::custom(0.7, 0.6);
    ///
    /// // Full customization
    /// let mut config = DiarizationConfig::custom(0.6, 0.5);
    /// config.min_duration_on = 0.3;  // Ignore segments shorter than 300ms
    /// config.median_window = 15;      // More smoothing
    /// ```
    pub fn custom(onset: f32, offset: f32) -> Self {
        Self {
            onset,
            offset,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.1,
            min_duration_off: 0.1,
            median_window: 11,
        }
    }
}

/// Speaker segment with start time, end time, and speaker ID
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    pub start: f32,
    pub end: f32,
    pub speaker_id: usize,
}

/// Streaming Sortformer v2 speaker diarization engine
pub struct Sortformer {
    session: Session,
    config: DiarizationConfig,
    // Streaming state. note that, Same way as Nemo
    spkcache: Array3<f32>,               // (1, 0..SPKCACHE_LEN, EMB_DIM)
    spkcache_preds: Option<Array3<f32>>, // (1, 0..SPKCACHE_LEN, NUM_SPEAKERS)
    fifo: Array3<f32>,                   // (1, 0..FIFO_LEN, EMB_DIM)
    fifo_preds: Array3<f32>,             // (1, 0..FIFO_LEN, NUM_SPEAKERS)
    mean_sil_emb: Array2<f32>,           // (1, EMB_DIM)
    n_sil_frames: usize,
    // Mel filterbank (cached)
    mel_basis: Array2<f32>,
}

impl Sortformer {
    /// a new Sortformer instance from ONNX model path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::with_config(model_path, None, DiarizationConfig::default())
    }

    /// Create with custom config
    pub fn with_config<P: AsRef<Path>>(
        model_path: P,
        execution_config: Option<ModelConfig>,
        config: DiarizationConfig,
    ) -> Result<Self> {
        let config_to_use = execution_config.unwrap_or_default();

        let session = config_to_use
            .apply_to_session_builder(Session::builder()?)?
            .commit_from_file(model_path.as_ref())?;

        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);

        let mut instance = Self {
            session,
            config,
            spkcache: Array3::zeros((1, 0, EMB_DIM)),
            spkcache_preds: None,
            fifo: Array3::zeros((1, 0, EMB_DIM)),
            fifo_preds: Array3::zeros((1, 0, NUM_SPEAKERS)),
            mean_sil_emb: Array2::zeros((1, EMB_DIM)),
            n_sil_frames: 0,
            mel_basis,
        };
        instance.reset_state();
        Ok(instance)
    }

    /// Reset streaming state
    pub fn reset_state(&mut self) {
        self.spkcache = Array3::zeros((1, 0, EMB_DIM));
        self.spkcache_preds = None;
        self.fifo = Array3::zeros((1, 0, EMB_DIM));
        self.fifo_preds = Array3::zeros((1, 0, NUM_SPEAKERS));
        self.mean_sil_emb = Array2::zeros((1, EMB_DIM));
        self.n_sil_frames = 0;
    }

    /// Main diarization entry point
    pub fn diarize(
        &mut self,
        mut audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<SpeakerSegment>> {
        // Resample if needed
        if sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Expected {} Hz, got {} Hz",
                SAMPLE_RATE, sample_rate
            )));
        }

        // Convert to mono
        if channels > 1 {
            audio = audio
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect();
        }

        // Reset state for new audio
        self.reset_state();

        // Extract mel features (B, T, D)
        let features = self.extract_mel_features(&audio);
        let total_frames = features.shape()[1];

        // Process in chunks
        let chunk_stride = CHUNK_LEN * SUBSAMPLING;
        let num_chunks = (total_frames + chunk_stride - 1) / chunk_stride;

        let mut all_chunk_preds = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_stride;
            let end = (start + chunk_stride).min(total_frames);
            let current_len = end - start;

            // Extract chunk features
            let mut chunk_feat = features.slice(s![.., start..end, ..]).to_owned();

            // Pad last chunk if needed
            if current_len < chunk_stride {
                let mut padded = Array3::zeros((1, chunk_stride, N_MELS));
                padded
                    .slice_mut(s![.., ..current_len, ..])
                    .assign(&chunk_feat);
                chunk_feat = padded;
            }

            // Run streaming update
            let chunk_preds = self.streaming_update(&chunk_feat, current_len)?;
            all_chunk_preds.push(chunk_preds);
        }

        // Concatenate all predictions
        let full_preds = Self::concat_predictions(&all_chunk_preds);

        // Apply median filtering
        let filtered_preds = if self.config.median_window > 1 {
            self.median_filter(&full_preds)
        } else {
            full_preds
        };

        // Binarize to segments
        let segments = self.binarize(&filtered_preds);

        Ok(segments)
    }

    /// NeMo's streaming_update with smart cache compression
    fn streaming_update(
        &mut self,
        chunk_feat: &Array3<f32>,
        current_len: usize,
    ) -> Result<Array2<f32>> {
        let spkcache_len = self.spkcache.shape()[1];
        let fifo_len = self.fifo.shape()[1];

        // Prepare inputs
        let chunk_lengths = Array1::from_vec(vec![current_len as i64]);
        let spkcache_lengths = Array1::from_vec(vec![spkcache_len as i64]);
        let fifo_lengths = Array1::from_vec(vec![fifo_len as i64]);

        // Prepare FIFO input
        let fifo_input = if fifo_len > 0 {
            self.fifo.clone()
        } else {
            Array3::zeros((1, 0, EMB_DIM))
        };

        // Prepare spkcache input (may be empty)
        let spkcache_input = if spkcache_len > 0 {
            self.spkcache.clone()
        } else {
            Array3::zeros((1, 0, EMB_DIM))
        };

        // Create input values
        let chunk_value = ort::value::Value::from_array(chunk_feat.clone())?;
        let chunk_lengths_value = ort::value::Value::from_array(chunk_lengths)?;
        let spkcache_value = ort::value::Value::from_array(spkcache_input)?;
        let spkcache_lengths_value = ort::value::Value::from_array(spkcache_lengths)?;
        let fifo_value = ort::value::Value::from_array(fifo_input)?;
        let fifo_lengths_value = ort::value::Value::from_array(fifo_lengths)?;

        // Run ONNX inference and extract all data in a block to release borrow
        let (preds, new_embs, chunk_len) = {
            let outputs = self.session.run(ort::inputs!(
                "chunk" => chunk_value,
                "chunk_lengths" => chunk_lengths_value,
                "spkcache" => spkcache_value,
                "spkcache_lengths" => spkcache_lengths_value,
                "fifo" => fifo_value,
                "fifo_lengths" => fifo_lengths_value
            ))?;

            // Extract outputs
            let (preds_shape, preds_data) = outputs["spkcache_fifo_chunk_preds"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract preds: {e}")))?;
            let (embs_shape, embs_data) = outputs["chunk_pre_encode_embs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract embs: {e}")))?;

            // Convert to ndarray
            let preds_dims = preds_shape.as_ref();
            let embs_dims = embs_shape.as_ref();

            let preds = Array3::from_shape_vec(
                (
                    preds_dims[0] as usize,
                    preds_dims[1] as usize,
                    preds_dims[2] as usize,
                ),
                preds_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape preds: {e}")))?;

            let new_embs = Array3::from_shape_vec(
                (
                    embs_dims[0] as usize,
                    embs_dims[1] as usize,
                    embs_dims[2] as usize,
                ),
                embs_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape embs: {e}")))?;

            // Calculate valid frames
            let valid_frames = (current_len + SUBSAMPLING - 1) / SUBSAMPLING;

            (preds, new_embs, valid_frames)
        };

        // Extract predictions for different parts
        let fifo_preds = if fifo_len > 0 {
            preds
                .slice(s![0, spkcache_len..spkcache_len + fifo_len, ..])
                .to_owned()
        } else {
            Array2::zeros((0, NUM_SPEAKERS))
        };

        let chunk_preds = preds
            .slice(s![
                0,
                spkcache_len + fifo_len..spkcache_len + fifo_len + chunk_len,
                ..
            ])
            .to_owned();
        let chunk_embs = new_embs.slice(s![0, ..chunk_len, ..]).to_owned();

        // Append chunk embeddings to FIFO
        self.fifo = Self::concat_axis1(&self.fifo, &chunk_embs.insert_axis(Axis(0)));

        // Update FIFO predictions
        if fifo_len > 0 {
            let combined = Self::concat_axis1_2d(&fifo_preds, &chunk_preds);
            self.fifo_preds = combined.insert_axis(Axis(0));
        } else {
            self.fifo_preds = chunk_preds.clone().insert_axis(Axis(0));
        }

        let fifo_len_after = self.fifo.shape()[1];

        // Move from FIFO to cache when FIFO exceeds limit
        if fifo_len_after > FIFO_LEN {
            let mut pop_out_len = SPKCACHE_UPDATE_PERIOD;
            pop_out_len = pop_out_len.max(chunk_len.saturating_sub(FIFO_LEN) + fifo_len);
            pop_out_len = pop_out_len.min(fifo_len_after);

            let pop_out_embs = self.fifo.slice(s![.., ..pop_out_len, ..]).to_owned();
            let pop_out_preds = self.fifo_preds.slice(s![.., ..pop_out_len, ..]).to_owned();

            // Update silence profile
            self.update_silence_profile(&pop_out_embs, &pop_out_preds);

            // Remove from FIFO
            self.fifo = self.fifo.slice(s![.., pop_out_len.., ..]).to_owned();
            self.fifo_preds = self.fifo_preds.slice(s![.., pop_out_len.., ..]).to_owned();

            // Append to cache
            self.spkcache = Self::concat_axis1(&self.spkcache, &pop_out_embs);

            if let Some(ref cache_preds) = self.spkcache_preds {
                self.spkcache_preds = Some(Self::concat_axis1(cache_preds, &pop_out_preds));
            }

            // Smart compression when cache exceeds limit
            if self.spkcache.shape()[1] > SPKCACHE_LEN {
                if self.spkcache_preds.is_none() {
                    // Initialize cache predictions from initial output
                    let initial_cache_preds = preds.slice(s![.., ..spkcache_len, ..]).to_owned();
                    let combined = Self::concat_axis1(&initial_cache_preds, &pop_out_preds);
                    self.spkcache_preds = Some(combined);
                }

                // Use smart compression
                self.compress_spkcache();
            }
        }

        Ok(chunk_preds)
    }

    /// Update mean silence embedding
    fn update_silence_profile(&mut self, embs: &Array3<f32>, preds: &Array3<f32>) {
        let preds_2d = preds.slice(s![0, .., ..]);

        for t in 0..preds_2d.shape()[0] {
            let sum: f32 = (0..NUM_SPEAKERS).map(|s| preds_2d[[t, s]]).sum();
            if sum < SIL_THRESHOLD {
                // This is a silence frame
                let emb = embs.slice(s![0, t, ..]);

                // Update running mean
                let old_sum: Vec<f32> = self
                    .mean_sil_emb
                    .slice(s![0, ..])
                    .iter()
                    .map(|&x| x * self.n_sil_frames as f32)
                    .collect();

                self.n_sil_frames += 1;

                for i in 0..EMB_DIM {
                    self.mean_sil_emb[[0, i]] = (old_sum[i] + emb[i]) / self.n_sil_frames as f32;
                }
            }
        }
    }

    /// Smart cache compression
    fn compress_spkcache(&mut self) {
        let cache_preds = match &self.spkcache_preds {
            Some(p) => p.clone(),
            None => return,
        };

        let n_frames = self.spkcache.shape()[1];
        let spkcache_len_per_spk = SPKCACHE_LEN / NUM_SPEAKERS - SPKCACHE_SIL_FRAMES_PER_SPK;
        let strong_boost_per_spk = (spkcache_len_per_spk as f32 * STRONG_BOOST_RATE) as usize;
        let weak_boost_per_spk = (spkcache_len_per_spk as f32 * WEAK_BOOST_RATE) as usize;
        let min_pos_scores_per_spk = (spkcache_len_per_spk as f32 * MIN_POS_SCORES_RATE) as usize;

        // Calculate quality scores
        let preds_2d = cache_preds.slice(s![0, .., ..]).to_owned();
        let mut scores = self.get_log_pred_scores(&preds_2d);

        // Disable low scores
        scores = self.disable_low_scores(&preds_2d, scores, min_pos_scores_per_spk);

        // Boost important frames
        scores = self.boost_topk_scores(scores, strong_boost_per_spk, 2.0);
        scores = self.boost_topk_scores(scores, weak_boost_per_spk, 1.0);

        // Add silence frames placeholder
        if SPKCACHE_SIL_FRAMES_PER_SPK > 0 {
            let mut padded = Array2::from_elem(
                (n_frames + SPKCACHE_SIL_FRAMES_PER_SPK, NUM_SPEAKERS),
                f32::NEG_INFINITY,
            );
            padded.slice_mut(s![..n_frames, ..]).assign(&scores);
            for i in n_frames..n_frames + SPKCACHE_SIL_FRAMES_PER_SPK {
                for j in 0..NUM_SPEAKERS {
                    padded[[i, j]] = f32::INFINITY;
                }
            }
            scores = padded;
        }

        // Select top frames
        let (topk_indices, is_disabled) = self.get_topk_indices(&scores, n_frames);

        // Gather embeddings
        let (new_embs, new_preds) = self.gather_spkcache(&topk_indices, &is_disabled);

        self.spkcache = new_embs;
        self.spkcache_preds = Some(new_preds);
    }

    /// Calculate quality scores
    fn get_log_pred_scores(&self, preds: &Array2<f32>) -> Array2<f32> {
        let mut scores = Array2::zeros(preds.dim());

        for t in 0..preds.shape()[0] {
            let mut log_1_probs_sum = 0.0f32;
            for s in 0..NUM_SPEAKERS {
                let p = preds[[t, s]].max(PRED_SCORE_THRESHOLD);
                let log_1_p = (1.0 - p).max(PRED_SCORE_THRESHOLD).ln();
                log_1_probs_sum += log_1_p;
            }

            for s in 0..NUM_SPEAKERS {
                let p = preds[[t, s]].max(PRED_SCORE_THRESHOLD);
                let log_p = p.ln();
                let log_1_p = (1.0 - p).max(PRED_SCORE_THRESHOLD).ln();
                scores[[t, s]] = log_p - log_1_p + log_1_probs_sum - 0.5f32.ln();
            }
        }

        scores
    }

    /// Disable non-speech and overlapped speech
    fn disable_low_scores(
        &self,
        preds: &Array2<f32>,
        mut scores: Array2<f32>,
        min_pos_scores_per_spk: usize,
    ) -> Array2<f32> {
        // Count positive scores per speaker
        let mut pos_count = vec![0usize; NUM_SPEAKERS];
        for t in 0..scores.shape()[0] {
            for s in 0..NUM_SPEAKERS {
                if scores[[t, s]] > 0.0 {
                    pos_count[s] += 1;
                }
            }
        }

        for t in 0..preds.shape()[0] {
            for s in 0..NUM_SPEAKERS {
                let is_speech = preds[[t, s]] > 0.5;

                if !is_speech {
                    scores[[t, s]] = f32::NEG_INFINITY;
                } else {
                    let is_pos = scores[[t, s]] > 0.0;
                    if !is_pos && pos_count[s] >= min_pos_scores_per_spk {
                        scores[[t, s]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        scores
    }

    /// Boost top K frames per speaker
    fn boost_topk_scores(
        &self,
        mut scores: Array2<f32>,
        n_boost_per_spk: usize,
        scale_factor: f32,
    ) -> Array2<f32> {
        for s in 0..NUM_SPEAKERS {
            // Get column for this speaker
            let col: Vec<(usize, f32)> = (0..scores.shape()[0])
                .map(|t| (t, scores[[t, s]]))
                .collect();

            // Sort by score descending
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Boost top K
            for i in 0..n_boost_per_spk.min(sorted.len()) {
                let t = sorted[i].0;
                if scores[[t, s]] != f32::NEG_INFINITY {
                    scores[[t, s]] -= scale_factor * 0.5f32.ln();
                }
            }
        }

        scores
    }

    /// Get indices of top frames
    fn get_topk_indices(
        &self,
        scores: &Array2<f32>,
        n_frames_no_sil: usize,
    ) -> (Vec<usize>, Vec<bool>) {
        let n_frames = scores.shape()[0];

        // Flatten scores as (S, T) then reshape to (S*T,)
        // This means we iterate: speaker 0 all times, then speaker 1 all times, etc.
        // flat_index = speaker * n_frames + time
        let mut flat_scores: Vec<(usize, f32)> = Vec::with_capacity(n_frames * NUM_SPEAKERS);
        for s in 0..NUM_SPEAKERS {
            for t in 0..n_frames {
                let flat_idx = s * n_frames + t;
                flat_scores.push((flat_idx, scores[[t, s]]));
            }
        }

        // Sort by score descending to get top-K
        flat_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top SPKCACHE_LEN and replace invalid scores with MAX_INDEX
        let mut topk_flat: Vec<usize> = flat_scores
            .iter()
            .take(SPKCACHE_LEN)
            .map(|(idx, score)| {
                if *score == f32::NEG_INFINITY {
                    MAX_INDEX
                } else {
                    *idx
                }
            })
            .collect();

        // Sort flat indices ascending (this puts MAX_INDEX at the end)
        topk_flat.sort();

        // Compute is_disabled and convert to frame indices
        let mut is_disabled = vec![false; SPKCACHE_LEN];
        let mut frame_indices = vec![0usize; SPKCACHE_LEN];

        for (i, &flat_idx) in topk_flat.iter().enumerate() {
            if flat_idx == MAX_INDEX {
                // Invalid entries are disabled
                is_disabled[i] = true;
                frame_indices[i] = 0; // We set disabled to 0
            } else {
                // convert to frame index
                let frame_idx = flat_idx % n_frames;

                // check if frame is beyond valid range
                if frame_idx >= n_frames_no_sil {
                    is_disabled[i] = true;
                    frame_indices[i] = 0; // same as abov: set disabled to 0
                } else {
                    frame_indices[i] = frame_idx;
                }
            }
        }

        (frame_indices, is_disabled)
    }

    /// Gather selected frames
    fn gather_spkcache(
        &self,
        indices: &[usize],
        is_disabled: &[bool],
    ) -> (Array3<f32>, Array3<f32>) {
        let mut new_embs = Array3::zeros((1, SPKCACHE_LEN, EMB_DIM));
        let mut new_preds = Array3::zeros((1, SPKCACHE_LEN, NUM_SPEAKERS));

        let cache_preds = self.spkcache_preds.as_ref().unwrap();

        for (i, (&idx, &disabled)) in indices.iter().zip(is_disabled.iter()).enumerate() {
            if i >= SPKCACHE_LEN {
                break;
            }

            if disabled {
                // Use silence embedding
                new_embs
                    .slice_mut(s![0, i, ..])
                    .assign(&self.mean_sil_emb.slice(s![0, ..]));
                // Predictions stay zero
            } else if idx < self.spkcache.shape()[1] {
                new_embs
                    .slice_mut(s![0, i, ..])
                    .assign(&self.spkcache.slice(s![0, idx, ..]));
                new_preds
                    .slice_mut(s![0, i, ..])
                    .assign(&cache_preds.slice(s![0, idx, ..]));
            }
        }

        (new_embs, new_preds)
    }

    /// Concatenate along axis 1 for 3D arrays
    fn concat_axis1(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
        if a.shape()[1] == 0 {
            return b.clone();
        }
        if b.shape()[1] == 0 {
            return a.clone();
        }
        ndarray::concatenate(Axis(1), &[a.view(), b.view()]).unwrap()
    }

    /// Concatenate along axis 0 for 2D arrays
    fn concat_axis1_2d(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        if a.shape()[0] == 0 {
            return b.clone();
        }
        if b.shape()[0] == 0 {
            return a.clone();
        }
        ndarray::concatenate(Axis(0), &[a.view(), b.view()]).unwrap()
    }

    /// Concatenate predictions
    fn concat_predictions(preds: &[Array2<f32>]) -> Array2<f32> {
        if preds.is_empty() {
            return Array2::zeros((0, NUM_SPEAKERS));
        }
        if preds.len() == 1 {
            return preds[0].clone();
        }

        let views: Vec<_> = preds.iter().map(|p| p.view()).collect();
        ndarray::concatenate(Axis(0), &views).unwrap()
    }

    /// Apply median filter to predictions
    fn median_filter(&self, preds: &Array2<f32>) -> Array2<f32> {
        let window = self.config.median_window;
        let half = window / 2;
        let mut filtered = preds.clone();

        for spk in 0..NUM_SPEAKERS {
            for t in 0..preds.shape()[0] {
                let start = t.saturating_sub(half);
                let end = (t + half + 1).min(preds.shape()[0]);

                let mut values: Vec<f32> = (start..end).map(|i| preds[[i, spk]]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                filtered[[t, spk]] = values[values.len() / 2];
            }
        }

        filtered
    }

    /// Binarize predictions to segments (padding applied during thresholding)
    fn binarize(&self, preds: &Array2<f32>) -> Vec<SpeakerSegment> {
        let mut segments = Vec::new();
        let num_frames = preds.shape()[0];

        for spk in 0..NUM_SPEAKERS {
            let mut in_seg = false;
            let mut seg_start = 0;
            let mut temp_segments = Vec::new();

            for t in 0..num_frames {
                let p = preds[[t, spk]];

                if p >= self.config.onset && !in_seg {
                    in_seg = true;
                    seg_start = t;
                } else if p < self.config.offset && in_seg {
                    in_seg = false;

                    // Apply padding during conversion
                    let start_t =
                        (seg_start as f32 * FRAME_DURATION - self.config.pad_onset).max(0.0);
                    let end_t = t as f32 * FRAME_DURATION + self.config.pad_offset;

                    if end_t - start_t >= self.config.min_duration_on {
                        temp_segments.push(SpeakerSegment {
                            start: start_t,
                            end: end_t,
                            speaker_id: spk,
                        });
                    }
                }
            }

            // Handle segment at end
            if in_seg {
                let start_t = (seg_start as f32 * FRAME_DURATION - self.config.pad_onset).max(0.0);
                let end_t = num_frames as f32 * FRAME_DURATION + self.config.pad_offset;

                if end_t - start_t >= self.config.min_duration_on {
                    temp_segments.push(SpeakerSegment {
                        start: start_t,
                        end: end_t,
                        speaker_id: spk,
                    });
                }
            }

            // Merge close segments (min_duration_off)
            if temp_segments.len() > 1 {
                let mut filtered = vec![temp_segments[0].clone()];
                for seg in temp_segments.into_iter().skip(1) {
                    let last = filtered.last_mut().unwrap();
                    let gap = seg.start - last.end;
                    if gap < self.config.min_duration_off {
                        last.end = seg.end; // Merge
                    } else {
                        filtered.push(seg);
                    }
                }
                segments.extend(filtered);
            } else {
                segments.extend(temp_segments);
            }
        }

        // Sort by start time
        segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
        segments
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(audio.len());
        result.push(audio[0]);
        for i in 1..audio.len() {
            result.push(audio[i] - PREEMPH * audio[i - 1]);
        }
        result
    }

    fn hann_window(window_length: usize) -> Vec<f32> {
        // Librosa uses periodic window (fftbins=True): divide by N, not N-1
        (0..window_length)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / window_length as f32).cos())
            .collect()
    }

    fn stft(audio: &[f32]) -> Array2<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // Create Hann window of length win_length, then zero-pad to n_fft (centered)
        // This is exactly what librosa does: util.pad_center(fft_window, size=n_fft)
        let hann = Self::hann_window(WIN_LENGTH);
        let win_offset = (N_FFT - WIN_LENGTH) / 2;
        let mut fft_window = vec![0.0f32; N_FFT];
        for i in 0..WIN_LENGTH {
            fft_window[win_offset + i] = hann[i];
        }

        // Pad signal for center=True (like librosa/torch.stft)
        // Padding is n_fft // 2 on each side
        let pad_amount = N_FFT / 2;
        let mut padded_audio = vec![0.0; pad_amount];
        padded_audio.extend_from_slice(audio);
        padded_audio.extend(vec![0.0; pad_amount]);

        let num_frames = (padded_audio.len() - N_FFT) / HOP_LENGTH + 1;
        let freq_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            let mut frame: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];

            // Extract n_fft samples and multiply by zero-padded window
            for i in 0..N_FFT {
                if start + i < padded_audio.len() {
                    frame[i] = Complex::new(padded_audio[start + i] * fft_window[i], 0.0);
                }
            }

            fft.process(&mut frame);
            for k in 0..freq_bins {
                let magnitude = frame[k].norm();
                // Power spectrum (magnitude^2) - NeMo uses mag_power=2.0
                spectrogram[[k, frame_idx]] = magnitude * magnitude;
            }
        }

        spectrogram
    }

    fn extract_mel_features(&self, audio: &[f32]) -> Array3<f32> {
        // 1. Add dither (small random noise to prevent log(0))
        // NeMo uses dither=1e-5, but for determinism we skip random noise
        // The log_zero_guard handles zero values

        // 2. Apply preemphasis (NeMo uses preemph=0.97)
        let preemphasized = Self::apply_preemphasis(audio);

        // 3. STFT
        let spectrogram = Self::stft(&preemphasized);

        // 4. Apply mel filterbank (with Slaney normalization)
        let mel_spec = self.mel_basis.dot(&spectrogram);

        // 5. Log with guard value (NeMo uses log_zero_guard_value = 2^-24)
        // NeMo uses normalize='NA' which means NO normalization
        let log_mel_spec = mel_spec.mapv(|x| (x + LOG_ZERO_GUARD).ln());

        let num_frames = log_mel_spec.shape()[1];
        let mut features = Array3::<f32>::zeros((1, num_frames, N_MELS));

        // Transpose to (batch, time, features) - NeMo outputs (B, D, T), model expects (B, T, D)
        for t in 0..num_frames {
            for m in 0..N_MELS {
                features[[0, t, m]] = log_mel_spec[[m, t]];
            }
        }

        features
    }
}
