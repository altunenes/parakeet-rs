//! NVIDIA Sortformer streaming speaker diarization (Nemotron-3 Diarization / Sortformer v3).
//!
//! Key features:
//! - Streaming inference with a FIFO queue and an arrival-order speaker cache
//! - Smart speaker cache compression (keeps informative frames, not just recent ones)
//! - Up to 8 speakers, IDs ordered by each speaker's first appearance
//! - Post-processing: optional median filtering, hysteresis thresholding
//!
//! Resolution: the model tracks the stream at 80ms frames (these drive the speaker cache) and
//! predicts speaker activity at 10ms frames. Everything this module returns is at 10ms.
//!
//! Latency: a streaming call emits nothing until `(chunk_len + right_context) * 80ms` of audio
//! has arrived. All streaming parameters are runtime settings; the exported graph has dynamic
//! time axes, so no re-export is needed to change them. See [`StreamingProfile`].
//!
//! Export the ONNX with `scripts/export_diar_sortformer.py`.
//! Reference: https://huggingface.co/nvidia/Nemotron-3-Diarization
//! Note, my stft code is adapted from: https://librosa.org/doc/main/generated/librosa.stft.html

use crate::error::{Error, Result};
use crate::execution::ModelConfig;
use crate::tensor_utils::extract_3d_f32;
use ndarray::{s, Array1, Array2, Array3, Axis};
use ort::session::Session;
use realfft::RealFftPlanner;
use std::path::Path;

// Model constants
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
const SAMPLE_RATE: usize = 16000;

// Streaming constants (defaults, overridden by ONNX metadata if present)
const CHUNK_LEN: usize = 340; // Frames per chunk (~27s at 80ms)
const FIFO_LEN: usize = 40; // FIFO buffer length
const SPKCACHE_LEN: usize = 264; // Speaker cache length
const RIGHT_CONTEXT: usize = 40; // Future frames for lookahead
const SPKCACHE_UPDATE_PERIOD: usize = 300; // Frames moved from FIFO to cache per update
pub const SUBSAMPLING: usize = 8; // Audio frames -> model frames
const UPSAMPLE_FACTOR: usize = 8; // Model (80ms) frames -> output (10ms) frames
const EMB_DIM: usize = 512; // Embedding dimension
pub const NUM_SPEAKERS: usize = 8; // Model supports 8 speakers
const FRAME_DURATION: f32 = 0.08; // 80ms per frame

// Cache compression params (from NeMo)
const SPKCACHE_SIL_FRAMES_PER_SPK: usize = 1;
const PRED_SCORE_THRESHOLD: f32 = 0.25;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;
const SCORES_BOOST_LATEST: f32 = 0.05;
const MAX_INDEX: usize = 99999;

/// Round to the nearest bfloat16 value. The checkpoint stores the STFT window and mel filterbank
/// in bf16 and NeMo runs with those values, so the features only match NeMo when ours do too.
fn to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)
}

/// Post-processing configuration for speaker diarization.
///
/// Controls how raw model predictions are converted into speaker segments.
///
/// # Parameters
/// - `onset`: Probability threshold to START a speaker segment (higher = more strict)
/// - `offset`: Probability threshold to END a speaker segment (lower = longer segments)
/// - `pad_onset`: Seconds to subtract from segment start times
/// - `pad_offset`: Seconds to add to segment end times
/// - `min_duration_on`: Minimum segment length in seconds (filters short blips)
/// - `min_duration_off`: Minimum gap between segments before merging
/// - `median_window`: Smoothing window in 10ms frames (odd number, `<= 1` disables it)
///
/// The default reproduces NeMo's `diarize()` for this model: a plain 0.5 threshold with no
/// padding, duration filtering, or smoothing. Use `custom(onset, offset)` to tune.
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
        Self {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            median_window: 1,
        }
    }
}

impl DiarizationConfig {
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
    /// config.median_window = 15;      // Smooth over 150ms
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

/// Streaming parameters, all in 80ms model frames.
///
/// Buffer latency is `(chunk_len + right_context) * 80ms`. The presets are NVIDIA's recommended
/// cfgs from the model card. note that any other combination also runs on the same ONNX.
///
/// | preset                | latency | spkcache | fifo | chunk | right_context | update |
/// |-----------------------|---------|----------|------|-------|---------------|--------|
/// | `offline()`           | 30.4 s  | 264      | 40   | 340   | 40            | 300    |
/// | `low_latency()`       | 1.04 s  | 264      | 264  | 9     | 4             | 222    |
/// | `very_low_latency()`  | 0.64 s  | 264      | 264  | 6     | 2             | 222    |
/// | `ultra_low_latency()` | 0.32 s  | 264      | 264  | 3     | 1             | 222    |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingProfile {
    /// Frames emitted per step.
    pub chunk_len: usize,
    /// Lookahead frames seen by the model but not emitted.
    pub right_context: usize,
    /// FIFO length between the chunk and the speaker cache.
    pub fifo_len: usize,
    /// Frames moved from the FIFO into the speaker cache per update.
    pub spkcache_update_period: usize,
    /// Speaker cache capacity (a multiple of [`NUM_SPEAKERS`]).
    pub spkcache_len: usize,
}

impl StreamingProfile {
    const fn nvidia(
        spkcache_len: usize,
        fifo_len: usize,
        chunk_len: usize,
        right_context: usize,
        spkcache_update_period: usize,
    ) -> Self {
        Self {
            chunk_len,
            right_context,
            fifo_len,
            spkcache_update_period,
            spkcache_len,
        }
    }

    /// Very high latency (offline), 30.4 s. What the exporter writes into the ONNX metadata.
    /// note that all those latency presets are coming from NVIDIA's recommended configs for this model (I did not invent them).
    pub const fn offline() -> Self {
        Self::nvidia(SPKCACHE_LEN, FIFO_LEN, CHUNK_LEN, RIGHT_CONTEXT, SPKCACHE_UPDATE_PERIOD)
    }

    /// 1.04 s.
    pub const fn low_latency() -> Self {
        Self::nvidia(264, 264, 9, 4, 222)
    }

    /// 0.64 s.
    pub const fn very_low_latency() -> Self {
        Self::nvidia(264, 264, 6, 2, 222)
    }

    /// 0.32 s
    pub const fn ultra_low_latency() -> Self {
        Self::nvidia(264, 264, 3, 1, 222)
    }
}

impl Default for StreamingProfile {
    fn default() -> Self {
        Self::offline()
    }
}

/// Speaker segment with start/end as sample offsets at 16 kHz, and speaker ID.
///
///
/// ```rust,ignore
/// let secs = seg.start as f64 / 16_000.0;
/// let nanos = seg.start as u64 * 1_000_000_000 / 16_000;
/// ```
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    /// Start position in samples at 16 kHz
    pub start: u64,
    /// End position in samples at 16 kHz
    pub end: u64,
    pub speaker_id: usize,
}

/// Raw per-frame speaker activity predictions (sigmoid outputs), one row per 10ms frame.
/// Used by the multitalker pipeline to derive speaker masks for the ASR encoder.
#[derive(Debug, Clone)]
pub struct RawDiarizationPredictions {
    /// Per-frame speaker activity probabilities, shape [num_frames, NUM_SPEAKERS].
    /// Values in [0.0, 1.0].
    pub predictions: Array2<f32>,
    /// Number of valid frames (may be <= predictions.nrows()).
    pub num_valid_frames: usize,
}

/// `(chunk mel frames, spkcache frames, fifo frames)` — the input dims that vary over a stream.
pub type StreamingWindow = (usize, usize, usize);

/// Supplies a session per streaming window, for graphs with fixed input shapes.
pub trait SessionRouter: Send + Sync {
    /// The session for this call, or `None` to use the one [`Sortformer`] owns.
    fn session_for(&mut self, window: StreamingWindow) -> Result<Option<&mut Session>>;

    /// Time spent inside ONNX for `window`.
    fn call_finished(&mut self, window: StreamingWindow, elapsed: std::time::Duration) {
        let _ = (window, elapsed);
    }

    /// Called from [`Sortformer::reset_state`].
    fn stream_reset(&mut self) {}
}

/// The window a graph is pinned to, or `None` if any input dim is symbolic.
pub fn graph_window(session: &Session) -> Option<StreamingWindow> {
    let dim = |name: &str| -> Option<usize> {
        let shape = session
            .inputs()
            .iter()
            .find(|i| i.name() == name)?
            .dtype()
            .tensor_shape()?
            .get(1)
            .copied()?;
        // A zero-length cache is a real window: every stream's first call has one.
        (shape >= 0).then_some(shape as usize)
    };
    Some((dim("chunk")?, dim("spkcache")?, dim("fifo")?))
}

/// Streaming Sortformer speaker diarization engine
pub struct Sortformer {
    session: Session,
    router: Option<Box<dyn SessionRouter>>,
    // The owned graph has fixed input shapes, so short chunks must be padded to full size.
    pinned: bool,
    config: DiarizationConfig,
    // Streaming constants (read from ONNX metadata, fallback to defaults)
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub right_context: usize,
    pub spkcache_update_period: usize,
    // Learned embedding for silence / unused cache slots (from ONNX metadata)
    sil_emb: Array1<f32>, // (EMB_DIM,)
    // Streaming state. note that, Same way as Nemo
    spkcache: Array3<f32>,               // (1, 0..spkcache_len, EMB_DIM)
    spkcache_preds: Option<Array3<f32>>, // (1, 0..spkcache_len, NUM_SPEAKERS), 80ms
    fifo: Array3<f32>,                   // (1, 0..fifo_len, EMB_DIM)
    fifo_preds: Array3<f32>,             // (1, 0..fifo_len, NUM_SPEAKERS), 80ms
    // Buffered streaming state (used by feed/flush)
    audio_buffer: Vec<f32>,
    elapsed_samples: usize,
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
        let session = config_to_use.build_session(model_path.as_ref())?;

        let has_output = |name: &str| session.outputs().iter().any(|o| o.name() == name);
        if !has_output("preds_diar") || !has_output("preds_hires") {
            return Err(Error::Config(
                "this ONNX has no preds_diar/preds_hires outputs; it looks like a Sortformer v2 \
                 export. Export Nemotron-3 Diarization with scripts/export_diar_sortformer.py"
                    .into(),
            ));
        }

        // Read streaming constants and the silence embedding from ONNX metadata.
        let (chunk_len, fifo_len, spkcache_len, right_context, spkcache_update_period, sil_emb) = {
            let metadata = session.metadata().ok();
            let get = |key: &str, default: usize| -> usize {
                metadata
                    .as_ref()
                    .and_then(|m| m.custom(key))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(default)
            };
            let sil_emb = metadata
                .as_ref()
                .and_then(|m| m.custom("learnable_sil_emb"))
                .map(|raw| {
                    raw.split(',')
                        .filter_map(|v| v.trim().parse().ok())
                        .collect::<Vec<f32>>()
                })
                .filter(|v| v.len() == EMB_DIM)
                .map(Array1::from_vec)
                .unwrap_or_else(|| Array1::zeros(EMB_DIM));
            (
                get("chunk_len", CHUNK_LEN),
                get("fifo_len", FIFO_LEN),
                get("spkcache_len", SPKCACHE_LEN),
                get("right_context", RIGHT_CONTEXT),
                get("spkcache_update_period", SPKCACHE_UPDATE_PERIOD),
                sil_emb,
            )
        };

        let mel_basis =
            crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE).mapv(to_bf16);

        let pinned = graph_window(&session).is_some();
        let mut instance = Self {
            session,
            router: None,
            pinned,
            config,
            chunk_len,
            fifo_len,
            spkcache_len,
            right_context,
            spkcache_update_period,
            sil_emb,
            spkcache: Array3::zeros((1, 0, EMB_DIM)),
            spkcache_preds: None,
            fifo: Array3::zeros((1, 0, EMB_DIM)),
            fifo_preds: Array3::zeros((1, 0, NUM_SPEAKERS)),
            audio_buffer: Vec::new(),
            elapsed_samples: 0,
            mel_basis,
        };
        instance.reset_state();
        Ok(instance)
    }

    /// Streaming latency in seconds: (chunk_len + right_context) * 80ms.
    /// eg. chunk_len=340, right_context=40 -> 30.4s
    pub fn latency(&self) -> f32 {
        (self.chunk_len + self.right_context) as f32 * FRAME_DURATION
    }

    /// The session this instance owns.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Route streaming calls through `router`; calls it declines run on this instance's session.
    pub fn set_session_router(&mut self, router: Box<dyn SessionRouter>) {
        self.router = Some(router);
    }

    /// The current streaming parameters.
    pub fn profile(&self) -> StreamingProfile {
        StreamingProfile {
            chunk_len: self.chunk_len,
            right_context: self.right_context,
            fifo_len: self.fifo_len,
            spkcache_update_period: self.spkcache_update_period,
            spkcache_len: self.spkcache_len,
        }
    }

    /// Validate and apply streaming parameters, then reset streaming state.
    pub fn set_profile(&mut self, profile: StreamingProfile) -> Result<()> {
        if profile.chunk_len == 0 {
            return Err(Error::Config("chunk_len must be > 0".into()));
        }
        if profile.spkcache_len == 0 || !profile.spkcache_len.is_multiple_of(NUM_SPEAKERS) {
            return Err(Error::Config(format!(
                "spkcache_len ({}) must be a positive multiple of {NUM_SPEAKERS}",
                profile.spkcache_len
            )));
        }
        self.chunk_len = profile.chunk_len;
        self.right_context = profile.right_context;
        self.fifo_len = profile.fifo_len;
        self.spkcache_update_period = profile.spkcache_update_period;
        self.spkcache_len = profile.spkcache_len;
        self.reset_state();
        Ok(())
    }

    /// Override the silence embedding, for an ONNX exported without `learnable_sil_emb` metadata.
    pub fn set_silence_embedding(&mut self, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != EMB_DIM {
            return Err(Error::Config(format!(
                "silence embedding must have {EMB_DIM} values, got {}",
                embedding.len()
            )));
        }
        self.sil_emb = Array1::from_vec(embedding);
        Ok(())
    }

    /// Log mel feats `(1, frames, 128)` as fed to the model.
    pub fn mel_features(&self, audio_16k_mono: &[f32]) -> Result<Array3<f32>> {
        self.extract_mel_features(audio_16k_mono)
    }

    /// Reset streaming state
    pub fn reset_state(&mut self) {
        if let Some(router) = self.router.as_mut() {
            router.stream_reset();
        }
        self.spkcache = Array3::zeros((1, 0, EMB_DIM));
        self.spkcache_preds = None;
        self.fifo = Array3::zeros((1, 0, EMB_DIM));
        self.fifo_preds = Array3::zeros((1, 0, NUM_SPEAKERS));
        self.audio_buffer.clear();
        self.elapsed_samples = 0;
    }

    /// Mel extraction and streaming inference, returning raw per-frame (10ms) speaker
    /// probabilities with no post-processing applied.
    ///
    /// # Returns
    /// `(predictions [num_frames, NUM_SPEAKERS], mono sample count)`
    pub fn predict_raw(
        &mut self,
        mut audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<(Array2<f32>, u64)> {
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

        // Extract mel features and run streaming inference
        let features = self.extract_mel_features(&audio)?;
        let full_preds = self.process_features(&features)?;

        Ok((full_preds, audio.len() as u64))
    }

    /// Median-smooth and binarize raw per-frame (10ms) speaker probabilities into segments,
    /// clipped to the audio length. Needs no session.
    pub fn post_process(
        config: &DiarizationConfig,
        preds: &Array2<f32>,
        n_audio_samples: u64,
    ) -> Vec<SpeakerSegment> {
        let filtered_owned;
        let filtered_preds = if config.median_window > 1 {
            filtered_owned = Self::median_filter(config, preds);
            &filtered_owned
        } else {
            preds
        };

        let mut segments = Self::binarize(config, filtered_preds);
        for seg in &mut segments {
            seg.end = seg.end.min(n_audio_samples);
        }
        segments.retain(|s| s.end > s.start);

        segments
    }

    /// Main diarization entry point
    pub fn diarize(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<SpeakerSegment>> {
        let (full_preds, n_audio_samples) = self.predict_raw(audio, sample_rate, channels)?;
        Ok(Self::post_process(
            &self.config,
            &full_preds,
            n_audio_samples,
        ))
    }

    /// Streaming diarization: process one audio chunk without resetting state.
    ///
    /// Unlike `diarize()`, this method preserves internal state (FIFO, speaker cache)
    /// across calls, enabling true streaming diarization.
    ///
    /// For full `right_context` benefit, buffer at least
    /// `(chunk_len + right_context) * 80ms` of audio before each call, then stride
    /// by `chunk_len * 80ms`. Shorter buffers still work (padded with zeros) but
    /// the lookahead sees silence instead of real future audio.
    ///
    /// # Arguments
    /// * `audio_16k_mono` - Audio chunk at 16kHz mono
    ///
    /// # Returns
    /// Speaker segments with sample offsets relative to this chunk (starting at 0)
    pub fn diarize_chunk(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerSegment>> {
        if audio_16k_mono.is_empty() {
            return Ok(vec![]);
        }

        let features = self.extract_mel_features(audio_16k_mono)?;
        let full_preds = self.process_features(&features)?;

        Ok(Self::post_process(
            &self.config,
            &full_preds,
            audio_16k_mono.len() as u64,
        ))
    }

    /// Streaming diarization returning raw predictions without post-processing.
    ///
    /// Unlike `diarize_chunk()`, this method returns the raw sigmoid outputs
    /// (per-10ms-frame speaker activity probabilities) without median filtering or
    /// binarisation. Used by the multitalker ASR pipeline to derive speaker
    /// masks for the encoder.
    ///
    /// # Arguments
    /// * `audio_16k_mono` - Audio chunk at 16kHz mono
    ///
    /// # Returns
    /// Raw predictions with shape [num_frames, NUM_SPEAKERS], values in [0.0, 1.0]
    pub fn diarize_chunk_raw(
        &mut self,
        audio_16k_mono: &[f32],
    ) -> Result<RawDiarizationPredictions> {
        if audio_16k_mono.is_empty() {
            return Ok(RawDiarizationPredictions {
                predictions: Array2::zeros((0, NUM_SPEAKERS)),
                num_valid_frames: 0,
            });
        }

        let features = self.extract_mel_features(audio_16k_mono)?;
        let full_preds = self.process_features(&features)?;
        let num_valid_frames = full_preds.nrows();

        Ok(RawDiarizationPredictions {
            predictions: full_preds,
            num_valid_frames,
        })
    }

    /// Feed audio samples for buffered streaming diarization.
    ///
    /// Buffers audio internally and runs inference only when enough data has
    /// accumulated for a full `(chunk_len + right_context)` window. Returns
    /// segments with **absolute** timestamps (accumulated across calls).
    ///
    /// Each window is binarized on its own, so a segment that crosses a window
    /// edge is returned as two segments.
    ///
    /// # Arguments
    /// * `audio_16k_mono` - Audio samples at 16kHz mono (any length)
    ///
    /// # Returns
    /// Speaker segments from any chunks that were ready, or empty vec if still buffering.
    pub fn feed(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerSegment>> {
        self.audio_buffer.extend_from_slice(audio_16k_mono);

        let feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING;
        let stride_samples = self.chunk_len * SUBSAMPLING * HOP_LENGTH;
        let feed_samples = (self.chunk_len + self.right_context) * SUBSAMPLING * HOP_LENGTH;
        let chunk_samples = stride_samples as u64;

        let mut all_segments = Vec::new();

        while self.audio_buffer.len() >= feed_samples {
            let window = &self.audio_buffer[..feed_samples];
            let features = self.extract_mel_features(window)?;
            // STFT center=True produces feed_size+1 mel frames from feed_samples audio,
            // so we always have enough frames: just slice to feed_size...
            let chunk_feat = features.slice(s![.., ..feed_size, ..]).to_owned();

            let chunk_preds = self.streaming_update(&chunk_feat, feed_size)?;

            // Binarize with absolute sample offset
            let sample_offset = self.elapsed_samples as u64;
            let mut segments = Self::post_process(&self.config, &chunk_preds, chunk_samples);
            for seg in &mut segments {
                seg.start += sample_offset;
                seg.end += sample_offset;
            }
            all_segments.extend(segments);

            // Advance: stride by chunk_len, keep right_context overlap
            self.audio_buffer.drain(..stride_samples);
            self.elapsed_samples += stride_samples;
        }

        Ok(all_segments)
    }

    /// Flush remaining buffered audio at end of stream.
    ///
    /// processes any leftover audio in the buffer with zero paddings.
    /// we call this once when the audio stream ends to get final segments.
    pub fn flush(&mut self) -> Result<Vec<SpeakerSegment>> {
        if self.audio_buffer.is_empty() {
            return Ok(vec![]);
        }

        let feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING;
        let remaining = std::mem::take(&mut self.audio_buffer);

        let features = self.extract_mel_features(&remaining)?;
        let total_mel = features.shape()[1];
        let current_len = total_mel.min(feed_size);

        let chunk_feat = self.pad_chunk(features.slice(s![.., ..current_len, ..]), feed_size);

        let chunk_preds = self.streaming_update(&chunk_feat, current_len)?;

        let sample_offset = self.elapsed_samples as u64;
        let mut segments =
            Self::post_process(&self.config, &chunk_preds, remaining.len() as u64);
        for seg in &mut segments {
            seg.start += sample_offset;
            seg.end += sample_offset;
        }

        self.elapsed_samples += remaining.len();

        Ok(segments)
    }

    /// Zero-pad a short (final) chunk to a whole number of 80ms frames, as NeMo's pre-encoder does.
    /// Padding further, to the full chunk size, changes the predictions of the real frames, so that
    /// is only done when the graph has fixed input shapes.
    fn pad_chunk(&self, chunk: ndarray::ArrayView3<f32>, feed_size: usize) -> Array3<f32> {
        let len = chunk.shape()[1];
        let target = if self.pinned {
            feed_size
        } else {
            len.next_multiple_of(SUBSAMPLING)
        };
        if len == target {
            return chunk.to_owned();
        }
        let mut padded = Array3::zeros((1, target, N_MELS));
        padded.slice_mut(s![.., ..len, ..]).assign(&chunk);
        padded
    }

    /// run streaming inference over mel features, returning concatenated per chunk predictions.
    /// note: this shared by `diarize`, `diarize_chunk`, and `diarize_chunk_raw`.
    fn process_features(&mut self, features: &Array3<f32>) -> Result<Array2<f32>> {
        let total_frames = features.shape()[1];
        let chunk_stride = self.chunk_len * SUBSAMPLING;
        let feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING;
        let num_chunks = total_frames.div_ceil(chunk_stride);

        let mut all_chunk_preds = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_stride;
            let end = (start + feed_size).min(total_frames);
            let current_len = end - start;

            let chunk_feat = self.pad_chunk(features.slice(s![.., start..end, ..]), feed_size);

            let chunk_preds = self.streaming_update(&chunk_feat, current_len)?;
            all_chunk_preds.push(chunk_preds);
        }

        let mut preds = Self::concat_predictions(&all_chunk_preds);
        // The last chunk rounds up to whole 80ms frames; trim to the audio's 10ms frame count.
        if preds.nrows() > total_frames {
            preds = preds.slice(s![..total_frames, ..]).to_owned();
        }
        Ok(preds)
    }

    /// NeMo's streaming_update with smart cache compression. The 80ms predictions update the
    /// FIFO and speaker cache; the chunk's 10ms predictions are returned.
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

        // Use empty arrays as fallbacks when lengths are zero (avoids cloning self fields)
        let empty_3d = Array3::<f32>::zeros((1, 0, EMB_DIM));
        let fifo_ref = if fifo_len > 0 { &self.fifo } else { &empty_3d };
        let spkcache_ref = if spkcache_len > 0 {
            &self.spkcache
        } else {
            &empty_3d
        };

        // Create borrowed tensor views instead of cloning arrays
        let chunk_value = ort::value::TensorRef::<f32>::from_array_view(chunk_feat.view())?;
        let chunk_lengths_value = ort::value::Value::from_array(chunk_lengths)?;
        let spkcache_value = ort::value::TensorRef::<f32>::from_array_view(spkcache_ref.view())?;
        let spkcache_lengths_value = ort::value::Value::from_array(spkcache_lengths)?;
        let fifo_value = ort::value::TensorRef::<f32>::from_array_view(fifo_ref.view())?;
        let fifo_lengths_value = ort::value::Value::from_array(fifo_lengths)?;

        let window = (chunk_feat.shape()[1], spkcache_len, fifo_len);
        let inference_start = self.router.is_some().then(std::time::Instant::now);
        let routed = match self.router.as_mut() {
            Some(router) => router.session_for(window)?,
            None => None,
        };

        // Run ONNX inference and extract all data in a block to release borrow
        let (preds_diar, preds_hires, new_embs) = {
            let session = match routed {
                Some(session) => session,
                None => &mut self.session,
            };
            let outputs = session.run(ort::inputs!(
                "chunk" => chunk_value,
                "chunk_lengths" => chunk_lengths_value,
                "spkcache" => spkcache_value,
                "spkcache_lengths" => spkcache_lengths_value,
                "fifo" => fifo_value,
                "fifo_lengths" => fifo_lengths_value
            ))?;

            (
                extract_3d_f32(&outputs["preds_diar"], "preds_diar")?,
                extract_3d_f32(&outputs["preds_hires"], "preds_hires")?,
                extract_3d_f32(&outputs["chunk_pre_encode_embs"], "chunk_pre_encode_embs")?,
            )
        };

        if let (Some(router), Some(started)) = (self.router.as_mut(), inference_start) {
            router.call_finished(window, started.elapsed());
        }

        // only keep chunk_len predictions/embeddings... right_context frames
        // participaded in attenttion (__providing lookahead__) but are discarded here.
        let valid_frames = current_len.div_ceil(SUBSAMPLING);
        let keep = self.chunk_len.min(valid_frames);

        // Extract 80ms predictions for different parts
        let fifo_preds = if fifo_len > 0 {
            preds_diar
                .slice(s![0, spkcache_len..spkcache_len + fifo_len, ..])
                .to_owned()
        } else {
            Array2::zeros((0, NUM_SPEAKERS))
        };
        let chunk_preds = preds_diar
            .slice(s![
                0,
                spkcache_len + fifo_len..spkcache_len + fifo_len + keep,
                ..
            ])
            .to_owned();
        let chunk_embs = new_embs.slice(s![0, ..keep, ..]).to_owned();

        // The same chunk at 10ms: each 80ms frame covers UPSAMPLE_FACTOR output frames.
        let hires_start = (spkcache_len + fifo_len) * UPSAMPLE_FACTOR;
        let chunk_preds_hires = preds_hires
            .slice(s![0, hires_start..hires_start + keep * UPSAMPLE_FACTOR, ..])
            .to_owned();

        // Append chunk embeddings to FIFO
        self.fifo = Self::concat_axis1(&self.fifo, &chunk_embs.insert_axis(Axis(0)));

        // Update FIFO predictions
        if fifo_len > 0 {
            let combined = Self::concat_axis1_2d(&fifo_preds, &chunk_preds);
            self.fifo_preds = combined.insert_axis(Axis(0));
        } else {
            self.fifo_preds = chunk_preds.insert_axis(Axis(0));
        }

        let fifo_len_after = self.fifo.shape()[1];

        // Move from FIFO to cache when FIFO exceeds limit. The pop length comes from
        // spkcache_update_period and the chunk length without right context, as in NeMo.
        if fifo_len_after > self.fifo_len {
            let mut pop_out_len = self.spkcache_update_period;
            // NeMo: chunk_len - self.fifo_len + fifo_len. Subtract last: keep < fifo_len here in
            // the low-latency presets, and clamping that difference first over-pops the FIFO.
            pop_out_len = pop_out_len.max((keep + fifo_len).saturating_sub(self.fifo_len));
            pop_out_len = pop_out_len.min(fifo_len_after);

            let pop_out_embs = self.fifo.slice(s![.., ..pop_out_len, ..]).to_owned();
            let pop_out_preds = self.fifo_preds.slice(s![.., ..pop_out_len, ..]).to_owned();

            // Remove from FIFO
            self.fifo = self.fifo.slice(s![.., pop_out_len.., ..]).to_owned();
            self.fifo_preds = self.fifo_preds.slice(s![.., pop_out_len.., ..]).to_owned();

            // Append to cache
            self.spkcache = Self::concat_axis1(&self.spkcache, &pop_out_embs);

            if let Some(ref cache_preds) = self.spkcache_preds {
                self.spkcache_preds = Some(Self::concat_axis1(cache_preds, &pop_out_preds));
            }

            // Smart compression when cache exceeds limit
            if self.spkcache.shape()[1] > self.spkcache_len {
                if self.spkcache_preds.is_none() {
                    // Initialize cache predictions from initial output
                    let initial_cache_preds =
                        preds_diar.slice(s![.., ..spkcache_len, ..]).to_owned();
                    let combined = Self::concat_axis1(&initial_cache_preds, &pop_out_preds);
                    self.spkcache_preds = Some(combined);
                }

                // Use smart compression
                self.compress_spkcache();
            }
        }

        Ok(chunk_preds_hires)
    }

    /// Smart cache compression
    fn compress_spkcache(&mut self) {
        let cache_preds = match &self.spkcache_preds {
            Some(p) => p.clone(),
            None => return,
        };

        let n_frames = self.spkcache.shape()[1];
        let per_spk = self.spkcache_len / NUM_SPEAKERS;
        if per_spk <= SPKCACHE_SIL_FRAMES_PER_SPK {
            // truncate if cache too small for compression
            self.spkcache = self.spkcache.slice(s![.., ..self.spkcache_len, ..]).to_owned();
            if let Some(ref p) = self.spkcache_preds {
                self.spkcache_preds = Some(p.slice(s![.., ..self.spkcache_len, ..]).to_owned());
            }
            return;
        }
        let spkcache_len_per_spk = per_spk - SPKCACHE_SIL_FRAMES_PER_SPK;
        let strong_boost_per_spk = (spkcache_len_per_spk as f32 * STRONG_BOOST_RATE) as usize;
        let weak_boost_per_spk = (spkcache_len_per_spk as f32 * WEAK_BOOST_RATE) as usize;
        let min_pos_scores_per_spk = (spkcache_len_per_spk as f32 * MIN_POS_SCORES_RATE) as usize;

        // Calculate quality scores
        let preds_2d = cache_preds.slice(s![0, .., ..]).to_owned();
        let mut scores = self.get_log_pred_scores(&preds_2d);

        // Disable low scores
        scores = self.disable_low_scores(&preds_2d, scores, min_pos_scores_per_spk);

        // Slightly favor frames appended since the last compression
        if SCORES_BOOST_LATEST > 0.0 {
            for t in self.spkcache_len.min(n_frames)..n_frames {
                for s in 0..NUM_SPEAKERS {
                    scores[[t, s]] += SCORES_BOOST_LATEST;
                }
            }
        }

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

        // As NeMo: log(clamp(p)) and log(clamp(1 - p)), each clamped separately on the raw p.
        for t in 0..preds.shape()[0] {
            let mut log_1_probs_sum = 0.0f32;
            for s in 0..NUM_SPEAKERS {
                log_1_probs_sum += (1.0 - preds[[t, s]]).max(PRED_SCORE_THRESHOLD).ln();
            }

            for s in 0..NUM_SPEAKERS {
                let log_p = preds[[t, s]].max(PRED_SCORE_THRESHOLD).ln();
                let log_1_p = (1.0 - preds[[t, s]]).max(PRED_SCORE_THRESHOLD).ln();
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
        let mut pos_count = [0usize; NUM_SPEAKERS];
        for t in 0..scores.shape()[0] {
            for s in 0..NUM_SPEAKERS {
                if scores[[t, s]] > 0.0 {
                    pos_count[s] += 1;
                }
            }
        }

        for t in 0..preds.shape()[0] {
            for s in 0..NUM_SPEAKERS {
                let non_speech = preds[[t, s]] <= 0.5;
                let low_overlap = scores[[t, s]] <= 0.0 && pos_count[s] >= min_pos_scores_per_spk;
                if non_speech || low_overlap {
                    scores[[t, s]] = f32::NEG_INFINITY;
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
            let mut sorted: Vec<(usize, f32)> = (0..scores.shape()[0])
                .map(|t| (t, scores[[t, s]]))
                .collect();

            // Sort by score descending
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Boost top K
            for &(t, _) in sorted.iter().take(n_boost_per_spk.min(sorted.len())) {
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

        // Take top spkcache_len and replace invalid scores with MAX_INDEX
        let mut topk_flat: Vec<usize> = flat_scores
            .iter()
            .take(self.spkcache_len)
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
        let mut is_disabled = vec![false; self.spkcache_len];
        let mut frame_indices = vec![0usize; self.spkcache_len];

        for (i, &flat_idx) in topk_flat.iter().enumerate() {
            if flat_idx == MAX_INDEX {
                // Invalid entries are disabled
                is_disabled[i] = true;
            } else {
                // convert to frame index
                let frame_idx = flat_idx % n_frames;

                // check if frame is beyond valid range
                if frame_idx >= n_frames_no_sil {
                    is_disabled[i] = true;
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
        let mut new_embs = Array3::zeros((1, self.spkcache_len, EMB_DIM));
        let mut new_preds = Array3::zeros((1, self.spkcache_len, NUM_SPEAKERS));

        let cache_preds = self.spkcache_preds.as_ref().unwrap();

        for (i, (&idx, &disabled)) in indices.iter().zip(is_disabled.iter()).enumerate() {
            if i >= self.spkcache_len {
                break;
            }

            if disabled {
                // Use silence embedding; predictions stay zero
                new_embs.slice_mut(s![0, i, ..]).assign(&self.sil_emb);
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
    fn median_filter(config: &DiarizationConfig, preds: &Array2<f32>) -> Array2<f32> {
        let window = config.median_window;
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

    /// Binarize 10ms predictions to segments, following NeMo's `binarization`: a segment starts
    /// when p > onset, ends when p < offset, then padding, min_duration_on and min_duration_off.
    fn binarize(config: &DiarizationConfig, preds: &Array2<f32>) -> Vec<SpeakerSegment> {
        let mut segments = Vec::new();
        let num_frames = preds.shape()[0];
        let frame_sec = FRAME_DURATION / UPSAMPLE_FACTOR as f32;

        let to_samples = |sec: f32| (sec * SAMPLE_RATE as f32).round().max(0.0) as u64;
        let pad_onset_samples = (config.pad_onset * SAMPLE_RATE as f32) as u64;
        let pad_offset_samples = (config.pad_offset * SAMPLE_RATE as f32) as u64;
        let min_dur_on_samples = to_samples(config.min_duration_on);
        // Padded neighbours can overlap; always merge those.
        let merge_gap = to_samples(config.min_duration_off)
            .max((pad_onset_samples + pad_offset_samples > 0) as u64);

        for spk in 0..NUM_SPEAKERS {
            // Hysteresis: runs of [start_frame, end_frame)
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut in_seg = false;
            let mut seg_start = 0;
            for t in 0..num_frames {
                let p = preds[[t, spk]];
                let active = if p > config.onset {
                    true
                } else if p < config.offset {
                    false
                } else {
                    in_seg
                };
                if active && !in_seg {
                    seg_start = t;
                } else if !active && in_seg {
                    runs.push((seg_start, t));
                }
                in_seg = active;
            }
            if in_seg {
                runs.push((seg_start, num_frames));
            }

            let mut temp_segments: Vec<SpeakerSegment> = Vec::new();
            for (start_f, end_f) in runs {
                let start = to_samples(start_f as f32 * frame_sec).saturating_sub(pad_onset_samples);
                let end = to_samples(end_f as f32 * frame_sec) + pad_offset_samples;
                if end <= start || end - start < min_dur_on_samples {
                    continue;
                }
                temp_segments.push(SpeakerSegment {
                    start,
                    end,
                    speaker_id: spk,
                });
            }

            // Merge close segments (min_duration_off)
            if merge_gap > 0 && temp_segments.len() > 1 {
                let mut merged = vec![temp_segments[0].clone()];
                for seg in temp_segments.into_iter().skip(1) {
                    let last = merged.last_mut().unwrap();
                    if seg.start.saturating_sub(last.end) < merge_gap {
                        last.end = last.end.max(seg.end); // Merge
                    } else {
                        merged.push(seg);
                    }
                }
                segments.extend(merged);
            } else {
                segments.extend(temp_segments);
            }
        }

        // Sort by start time
        segments.sort_by_key(|s| (s.start, s.speaker_id));
        segments
    }

    fn hann_window(window_length: usize) -> Vec<f32> {
        // NeMo uses torch.hann_window(periodic=False): divide by N-1, not N
        let n = (window_length - 1) as f64;
        (0..window_length)
            .map(|i| to_bf16((0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n).cos()) as f32))
            .collect()
    }

    fn stft(audio: &[f32]) -> Result<Array2<f32>> {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(N_FFT);

        // Create Hann window of length win_length, then zero-pad to n_fft (centered)
        // This is exactly what librosa does: util.pad_center(fft_window, size=n_fft)
        let hann = Self::hann_window(WIN_LENGTH);
        let win_offset = (N_FFT - WIN_LENGTH) / 2;
        let mut fft_window = vec![0.0f32; N_FFT];
        fft_window[win_offset..(WIN_LENGTH + win_offset)].copy_from_slice(&hann[..WIN_LENGTH]);

        // Pad signal for center=True (like librosa/torch.stft)
        // Padding is n_fft // 2 on each side
        let pad_amount = N_FFT / 2;
        let mut padded_audio = vec![0.0; pad_amount];
        padded_audio.extend_from_slice(audio);
        padded_audio.extend(vec![0.0; pad_amount]);

        let num_frames = (padded_audio.len() - N_FFT) / HOP_LENGTH + 1;
        let freq_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

        let mut input = vec![0.0f32; N_FFT];
        let mut output = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;

            // Extract n_fft samples and multiply by zero-padded window
            for i in 0..N_FFT {
                input[i] = if start + i < padded_audio.len() {
                    padded_audio[start + i] * fft_window[i]
                } else {
                    0.0
                };
            }

            r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
                .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

            for k in 0..freq_bins {
                // Power spectrum (magnitude^2) - NeMo uses mag_power=2.0
                spectrogram[[k, frame_idx]] = output[k].norm_sqr();
            }
        }

        Ok(spectrogram)
    }

    fn extract_mel_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        // 1. Add dither (small random noise to prevent log(0))
        // NeMo uses dither=1e-5, but for determinism we skip random noise
        // The log_zero_guard handles zero values

        // 2. Apply preemphasis (NeMo uses preemph=0.97)
        let preemphasized = crate::audio::apply_preemphasis(audio, PREEMPH);

        // 3. STFT
        let spectrogram = Self::stft(&preemphasized)?;

        // 4. Apply mel filterbank (with Slaney normalization)
        let mel_spec = self.mel_basis.dot(&spectrogram);

        // 5. Log with guard value (NeMo uses log_zero_guard_value = 2^-24)
        // NeMo uses normalize='NA' which means NO normalization
        let log_mel_spec = mel_spec.mapv(|x| (x + LOG_ZERO_GUARD).ln());

        // Transpose to (batch, time, features) - NeMo outputs (B, D, T), model expects (B, T, D)
        Ok(log_mel_spec.t().to_owned().insert_axis(Axis(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_expected_bin() {
        // 1kHz sine at 16kHz sample rate, 1 second
        let audio = sine_wave(1000.0, SAMPLE_RATE, SAMPLE_RATE);
        let spec = Sortformer::stft(&audio).unwrap();

        // Expected bin: 1000 * N_FFT / SAMPLE_RATE = 1000 * 512 / 16000 = 32
        let expected_bin = 32;
        let freq_bins = N_FFT / 2 + 1;
        let num_frames = spec.shape()[1];

        let mut correct_frames = 0;
        for frame in 2..num_frames.saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..freq_bins {
                if spec[[bin, frame]] > max_power {
                    max_power = spec[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct_frames += 1;
            }
        }

        let interior_frames = num_frames.saturating_sub(4);
        assert!(
            correct_frames > interior_frames / 2,
            "Expected bin {expected_bin} to dominate, but only {correct_frames}/{interior_frames}"
        );
    }

    #[test]
    fn stft_output_shape_is_correct() {
        let audio = vec![0.0f32; SAMPLE_RATE]; // 1 second
        let spec = Sortformer::stft(&audio).unwrap();

        let freq_bins = N_FFT / 2 + 1;
        assert_eq!(spec.shape()[0], freq_bins);
        assert!(spec.shape()[1] > 0);
    }

    #[test]
    fn binarize_matches_hysteresis() {
        // speaker 0 active for frames 10..30 (10ms frames); onset/offset 0.5
        let mut preds = Array2::zeros((50, NUM_SPEAKERS));
        for t in 10..30 {
            preds[[t, 0]] = 0.9;
        }
        let segs = Sortformer::post_process(&DiarizationConfig::default(), &preds, 1_000_000);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].speaker_id, 0);
        assert_eq!(segs[0].start, 10 * 160);
        assert_eq!(segs[0].end, 30 * 160);
    }
}
