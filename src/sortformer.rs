//! NVIDIA Sortformer speaker diarization
//!
//! This module implements NVIDIA's Sortformer model for speaker diarization.
//!
//! Reference: https://github.com/NVIDIA/NeMo
//!
//! Key differences from our standard mel extraction src/audio.rs  (based on the https://github.com/NVIDIA-NeMo/NeMo/blob/559cc979d09f12734bc4868bf13602ff51e1ebd5/nemo/collections/asr/parts/preprocessing/features.py#L246):
//! - Audio normalization BEFORE processing: audio / (max(abs(audio)) + 0.001)
//! - Mel range: 0 to 8000 Hz
//! - Log guard: 2^-24
//! - seq_len calculation: (audio_len + n_fft - n_fft) / hop_length
//! - Normalize over first seq_len frames
//! - Std formula: sqrt(sum / (seq_len - 1)) (N-1 denominator)
//! - Mask frames beyond seq_len to 0.0
//! - Pad time dimension to multiple of 16

use crate::error::{Error, Result};
use ndarray::{s, Array1, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::path::Path;

/// Speaker segment with start time, end time, and speaker ID
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    pub start: f32,
    pub end: f32,
    pub speaker_id: usize,
}

/// Sortformer configuration matching NeMo's preprocessing
/// https://huggingface.co/nvidia/diar_sortformer_4spk-v1/blob/main/processor_config.json
/// https://huggingface.co/nvidia/diar_sortformer_4spk-v1/blob/main/config.json
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960464478e-8;
const PAD_TO: usize = 16;
const SAMPLE_RATE: usize = 16000;
const FMAX: f32 = 8000.0;

/// Sortformer speaker diarization engine
pub struct Sortformer {
    session: Session,
}

impl Sortformer {
    /// a new Sortformer instance from ONNX model path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(audio.len());
        result.push(audio[0]);

        for i in 1..audio.len() {
            result.push(audio[i] - PREEMPH * audio[i - 1]);
        }

        result
    }

    /// Hann window (periodic=False trying to match with torch...)
    fn hann_window(window_length: usize) -> Vec<f32> {
        (0..window_length)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
            .collect()
    }

    /// STFT with center padding
    fn stft(audio: &[f32]) -> Array2<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let window = Self::hann_window(WIN_LENGTH);

        // Center padding: pad_amount = n_fft // 2 * 2 = n_fft
        let pad_amount = N_FFT;
        let mut padded_audio = vec![0.0; pad_amount / 2];
        padded_audio.extend_from_slice(audio);
        padded_audio.extend(vec![0.0; pad_amount / 2]);

        let num_frames = (padded_audio.len() - N_FFT) / HOP_LENGTH + 1;
        let freq_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;

            let mut frame: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
            for i in 0..WIN_LENGTH.min(padded_audio.len() - start) {
                frame[i] = Complex::new(padded_audio[start + i] * window[i], 0.0);
            }

            fft.process(&mut frame);

            for k in 0..freq_bins {
                let magnitude = frame[k].norm();
                spectrogram[[k, frame_idx]] = magnitude * magnitude;
            }
        }

        spectrogram
    }
    // utils based on the: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py
    /// Convert Hz to Mel scale
    fn hz_to_mel(freq: f32) -> f32 {
        2595.0 * (1.0 + freq / 700.0).log10()
    }

    /// Convert Mel to Hz scale
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Create mel filterbank (Slaney normalization, 0-8kHz range)
    fn create_mel_filterbank() -> Array2<f32> {
        let freq_bins = N_FFT / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((N_MELS, freq_bins));

        let min_mel = Self::hz_to_mel(0.0);
        let max_mel = Self::hz_to_mel(FMAX); // 8kHz max, not sr/2

        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| Self::mel_to_hz(min_mel + (max_mel - min_mel) * i as f32 / (N_MELS + 1) as f32))
            .collect();

        let freq_bin_width = SAMPLE_RATE as f32 / N_FFT as f32;

        for mel_idx in 0..N_MELS {
            let left = mel_points[mel_idx];
            let center = mel_points[mel_idx + 1];
            let right = mel_points[mel_idx + 2];

            for freq_idx in 0..freq_bins {
                let freq = freq_idx as f32 * freq_bin_width;

                if freq >= left && freq <= center {
                    filterbank[[mel_idx, freq_idx]] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    filterbank[[mel_idx, freq_idx]] = (right - freq) / (right - center);
                }
            }
        }

        // Slaney normalization: normalize each filter to have area 2.0 / (mel_points[i+2] - mel_points[i])
        for mel_idx in 0..N_MELS {
            let bandwidth = mel_points[mel_idx + 2] - mel_points[mel_idx];
            let norm_factor = 2.0 / bandwidth;
            for freq_idx in 0..freq_bins {
                filterbank[[mel_idx, freq_idx]] *= norm_factor;
            }
        }

        filterbank
    }

    /// Extract Sortformer-compatible mel features
    fn extract_mel_features(audio: &[f32]) -> (Array3<f32>, i64) {
        // we need to normalize audio waveform BEFORE processing to match offical example
        let max_val = audio
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let audio_normalized: Vec<f32> = audio.iter().map(|x| x / (max_val + 0.001)).collect();

        let audio_len = audio_normalized.len();

        // Calculate seq_len (NeMo's way)
        let pad_amount = N_FFT;
        let seq_len = (audio_len + pad_amount - N_FFT) / HOP_LENGTH;

        // Preemphasis
        let audio_pre = Self::apply_preemphasis(&audio_normalized);

        // STFT
        let spec = Self::stft(&audio_pre);

        // Mel filterbank
        let mel_basis = Self::create_mel_filterbank();
        // (n_mels, time)
        let mel = mel_basis.dot(&spec);

        // Log
        let mel = mel.mapv(|x| (x + LOG_ZERO_GUARD).ln());

        // Transpose to (time, n_mels)
        let mut mel = mel.t().to_owned();

        // Per-feature normalization (only over valid frames)
        let max_time = mel.shape()[0];

        for feat_idx in 0..N_MELS {
            // Only use first seq_len frames for statistics
            let valid_frames = mel.slice(s![..seq_len, feat_idx]);

            let mean = valid_frames.sum() / seq_len as f32;

            let variance = valid_frames
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / (seq_len - 1) as f32; // N-1 denominator!

            let std = variance.sqrt() + 1e-5;

            // Apply normalization to ALL frames
            let mut column = mel.column_mut(feat_idx);
            for val in column.iter_mut() {
                *val = (*val - mean) / std;
            }
        }

        // Mask frames beyond seq_len to 0
        if seq_len < max_time {
            mel.slice_mut(s![seq_len.., ..]).fill(0.0);
        }

        // Pad to multiple of 16
        let pad_amt = mel.shape()[0] % PAD_TO;
        let mel = if pad_amt != 0 {
            let padding_rows = PAD_TO - pad_amt;
            let padding = Array2::<f32>::zeros((padding_rows, N_MELS));
            ndarray::concatenate(ndarray::Axis(0), &[mel.view(), padding.view()]).unwrap()
        } else {
            mel
        };

        // Reshape to (1, n_mels, time) for ONNX
        let mel = mel.t().to_owned().insert_axis(ndarray::Axis(0));

        (mel, seq_len as i64)
    }

    /// Extract speaker segments from predictions
    fn extract_segments(predictions: &Array2<f32>, num_speakers: usize) -> Vec<SpeakerSegment> {
        let mut segments = Vec::new();
        const THRESHOLD: f32 = 0.5;
        const FRAME_DURATION: f32 = 0.08;

        for speaker in 0..num_speakers {
            let mut in_segment = false;
            let mut start_frame = 0;

            for frame in 0..predictions.shape()[0] {
                let active = predictions[[frame, speaker]] > THRESHOLD;

                if active && !in_segment {
                    start_frame = frame;
                    in_segment = true;
                } else if !active && in_segment {
                    segments.push(SpeakerSegment {
                        start: start_frame as f32 * FRAME_DURATION,
                        end: frame as f32 * FRAME_DURATION,
                        speaker_id: speaker,
                    });
                    in_segment = false;
                }
            }

            if in_segment {
                segments.push(SpeakerSegment {
                    start: start_frame as f32 * FRAME_DURATION,
                    end: predictions.shape()[0] as f32 * FRAME_DURATION,
                    speaker_id: speaker,
                });
            }
        }

        // Sort by start time
        segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());

        segments
    }

    /// Perform speaker diarization on audio samples
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples as f32 values
    /// * `sample_rate` - Sample rate in Hz (must be 16000)
    /// * `channels` - Number of audio channels
    ///
    /// # Returns
    ///
    /// Vector of speaker segments with start time, end time, and speaker ID (0-3)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hound;
    ///
    /// let mut reader = hound::WavReader::open("audio.wav")?;
    /// let spec = reader.spec();
    /// let audio: Vec<f32> = reader.samples::<i16>()
    ///     .map(|s| s.unwrap() as f32 / 32768.0)
    ///     .collect();
    ///
    /// // Perform diarization
    /// let mut sortformer = Sortformer::new("model.onnx")?;
    /// let segments = sortformer.diarize(audio, spec.sample_rate, spec.channels)?;
    /// ```
    pub fn diarize(
        &mut self,
        mut audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<SpeakerSegment>> {
        // Validate sample rate
        if sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Audio must be 16kHz, got {}Hz",
                sample_rate
            )));
        }

        // Convert to mono if needed
        if channels > 1 {
            audio = audio
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect();
        }

        // Extract mel features
        let (mel_features, mel_lengths) = Self::extract_mel_features(&audio);

        // Run ONNX inference
        let mel_array = mel_features.as_standard_layout().to_owned();
        let lengths_array = Array1::from_vec(vec![mel_lengths]);

        let mel_value = ort::value::Value::from_array(mel_array)?;
        let lengths_value = ort::value::Value::from_array(lengths_array)?;

        let outputs = self.session.run(ort::inputs![
            "mel_features" => mel_value,
            "mel_lengths" => lengths_value
        ])?;

        // Extract predictions (frames, 4) - get first output
        let predictions_value = outputs.iter().next().ok_or_else(|| {
            Error::Model("Model produced no outputs".to_string())
        })?.1;
        let (shape, data) = predictions_value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract predictions: {e}")))?;

        let shape_dims = shape.as_ref();
        if shape_dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D predictions, got shape: {shape_dims:?}"
            )));
        }

        let batch_size = shape_dims[0] as usize;
        let num_frames = shape_dims[1] as usize;
        let num_speakers = shape_dims[2] as usize;

        if batch_size != 1 {
            return Err(Error::Model(format!(
                "Expected batch size 1, got {batch_size}"
            )));
        }

        // Create 2D array (frames, speakers)
        let predictions = Array2::from_shape_vec((num_frames, num_speakers), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create predictions array: {e}")))?;

        // Extract segments
        let segments = Self::extract_segments(&predictions, 4);

        Ok(segments)
    }
}
