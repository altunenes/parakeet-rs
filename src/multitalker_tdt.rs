//! Per-speaker transcripts, overlaps included: parakeet-tdt-0.6b-v3 with speaker kernels.
//! Sortformer says who speaks when, then TDT runs once per speaker in ~20 s chunks.

use crate::decoder::TimedToken;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::parakeet_tdt::ParakeetTDT;
use crate::sortformer::{DiarizationConfig, Sortformer, SpeakerSegment};
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use std::path::Path;

const SAMPLE_RATE: usize = 16_000;
/// 80 ms.
const FRAME_SAMPLES: usize = 1280;
/// How far a cut may move to land in a pause.
const CUT_REACH: usize = 3 * SAMPLE_RATE;
/// 40 ms.
const CUT_STEP: usize = 640;

#[derive(Debug, Clone)]
pub struct MultitalkerTDTConfig {
    /// Seconds.
    pub chunk_seconds: f32,
    /// Less speech than this in a chunk (s): speaker skipped there.
    pub min_speech_seconds: f32,
    /// A longer gap between words starts a new utterance (s).
    pub max_gap_seconds: f32,
}

impl Default for MultitalkerTDTConfig {
    fn default() -> Self {
        Self {
            chunk_seconds: 20.0,
            min_speech_seconds: 0.5,
            max_gap_seconds: 1.0,
        }
    }
}

/// One speaker's consecutive words. Times in seconds.
#[derive(Debug, Clone)]
pub struct SpeakerUtterance {
    pub speaker_id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub words: Vec<TimedToken>,
}

pub struct MultitalkerTDT {
    model: ParakeetTDT,
    diarizer: Sortformer,
    config: MultitalkerTDTConfig,
}

impl MultitalkerTDT {
    /// `model_dir`: the multitalker TDT export. `diarizer_path`: Nemotron-3 Diarization ONNX.
    pub fn from_pretrained<P: AsRef<Path>, D: AsRef<Path>>(
        model_dir: P,
        diarizer_path: D,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model = ParakeetTDT::from_pretrained(model_dir, exec_config.clone())?;
        let diarizer =
            Sortformer::with_config(diarizer_path, exec_config, DiarizationConfig::default())?;
        Ok(Self::new(model, diarizer))
    }

    pub fn new(model: ParakeetTDT, diarizer: Sortformer) -> Self {
        Self {
            model,
            diarizer,
            config: MultitalkerTDTConfig::default(),
        }
    }

    pub fn with_config(mut self, config: MultitalkerTDTConfig) -> Self {
        self.config = config;
        self
    }

    /// 16 kHz only. Overlapping speech gives overlapping utterances.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<SpeakerUtterance>> {
        if sample_rate as usize != SAMPLE_RATE {
            return Err(Error::Audio(format!(
                "multitalker TDT needs 16 kHz audio, got {sample_rate} Hz"
            )));
        }
        let channels = channels.max(1) as usize;
        let mono: Vec<f32> = audio
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        let segments = self.diarizer.diarize(mono.clone(), sample_rate, 1)?;
        self.transcribe_with_segments(&mono, &segments)
    }

    /// Bring your own diarization. 16 kHz mono.
    pub fn transcribe_with_segments(
        &mut self,
        audio: &[f32],
        segments: &[SpeakerSegment],
    ) -> Result<Vec<SpeakerUtterance>> {
        let mut speakers: Vec<usize> = segments.iter().map(|s| s.speaker_id).collect();
        speakers.sort_unstable();
        speakers.dedup();

        let chunk = (self.config.chunk_seconds * SAMPLE_RATE as f32) as usize;
        let min_frames = self.config.min_speech_seconds * SAMPLE_RATE as f32 / FRAME_SAMPLES as f32;
        let edges = chunk_edges(audio.len(), segments, chunk);
        let mut utterances = Vec::new();
        for pair in edges.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            if end - start < SAMPLE_RATE / 2 {
                continue;
            }
            let masks: Vec<Vec<f32>> = speakers
                .iter()
                .map(|&speaker| activity(segments, speaker, start, end))
                .collect();
            for (i, &speaker) in speakers.iter().enumerate() {
                if masks[i].iter().sum::<f32>() < min_frames {
                    continue;
                }
                let others = (0..masks[i].len())
                    .map(|f| {
                        let other = masks.iter().enumerate().filter(|(j, _)| *j != i);
                        other.map(|(_, m)| m[f]).fold(0.0, f32::max)
                    })
                    .collect();
                self.model
                    .set_speaker_targets(Some((masks[i].clone(), others)));
                let result = self.model.transcribe_samples(
                    audio[start..end].to_vec(),
                    SAMPLE_RATE as u32,
                    1,
                    Some(TimestampMode::Words),
                );
                self.model.set_speaker_targets(None);
                let offset = start as f32 / SAMPLE_RATE as f32;
                let words = result?.tokens.into_iter().map(|w| TimedToken {
                    start: w.start + offset,
                    end: w.end + offset,
                    text: w.text,
                });
                utterances.extend(group(speaker, words, self.config.max_gap_seconds));
            }
        }
        utterances.sort_by(|a, b| a.start.total_cmp(&b.start));
        Ok(utterances)
    }
}

/// A cut every `chunk` samples, moved to the nearest pause within 3 s.
fn chunk_edges(len: usize, segments: &[SpeakerSegment], chunk: usize) -> Vec<usize> {
    let busy = |t: usize| {
        segments
            .iter()
            .any(|s| (s.start as usize) < t && t < s.end as usize)
    };
    let mut edges = vec![0];
    while len - edges[edges.len() - 1] > chunk {
        let last = edges[edges.len() - 1];
        let target = last + chunk;
        let lowest = target.saturating_sub(CUT_REACH).max(last + CUT_STEP);
        let highest = (target + CUT_REACH).min(len.saturating_sub(SAMPLE_RATE));
        let cut = (lowest..highest)
            .step_by(CUT_STEP)
            .filter(|&t| !busy(t))
            .min_by_key(|&t| t.abs_diff(target))
            .unwrap_or(target);
        edges.push(cut);
    }
    edges.push(len);
    edges
}

/// 1.0 where `speaker` talks, per 80 ms frame of `start..end`.
fn activity(segments: &[SpeakerSegment], speaker: usize, start: usize, end: usize) -> Vec<f32> {
    let frames = (end - start).div_ceil(FRAME_SAMPLES);
    let mut mask = vec![0.0f32; frames];
    for seg in segments.iter().filter(|s| s.speaker_id == speaker) {
        let first = (seg.start as usize).saturating_sub(start) / FRAME_SAMPLES;
        let last = (seg.end as usize)
            .saturating_sub(start)
            .div_ceil(FRAME_SAMPLES);
        for value in mask.iter_mut().take(last.min(frames)).skip(first) {
            *value = 1.0;
        }
    }
    mask
}

/// Words to utterances; punctuation sticks to the word before.
fn group(
    speaker: usize,
    words: impl IntoIterator<Item = TimedToken>,
    max_gap: f32,
) -> Vec<SpeakerUtterance> {
    let mut out: Vec<SpeakerUtterance> = Vec::new();
    for word in words {
        let punctuation = word.text.chars().all(|c| c.is_ascii_punctuation());
        match out.last_mut() {
            Some(u) if punctuation || word.start - u.end <= max_gap => {
                if !punctuation {
                    u.text.push(' ');
                    u.end = word.end;
                }
                u.text.push_str(&word.text);
                u.words.push(word);
            }
            _ if punctuation => {}
            _ => out.push(SpeakerUtterance {
                speaker_id: speaker,
                start: word.start,
                end: word.end,
                text: word.text.clone(),
                words: vec![word],
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, speaker_id: usize) -> SpeakerSegment {
        let s = |t: f32| (t * SAMPLE_RATE as f32) as u64;
        SpeakerSegment {
            start: s(start),
            end: s(end),
            speaker_id,
        }
    }

    fn word(text: &str, start: f32, end: f32) -> TimedToken {
        TimedToken {
            text: text.into(),
            start,
            end,
        }
    }

    #[test]
    fn cuts_move_to_pauses() {
        let sr = SAMPLE_RATE;
        // Speech until 21.5 s, then a pause: the 20 s cut moves into it.
        let segments = [seg(0.0, 21.5, 0), seg(22.0, 45.0, 1)];
        let edges = chunk_edges(50 * sr, &segments, 20 * sr);
        assert_eq!(edges[0], 0);
        assert!(
            edges[1] > 21 * sr + sr / 2 && edges[1] < 22 * sr,
            "{edges:?}"
        );
        assert_eq!(*edges.last().unwrap(), 50 * sr);
        // No pause within reach: cut at the target.
        let edges = chunk_edges(50 * sr, &[seg(0.0, 50.0, 0)], 20 * sr);
        assert_eq!(edges, vec![0, 20 * sr, 40 * sr, 50 * sr]);
        // Short audio: one chunk.
        assert_eq!(chunk_edges(5 * sr, &[], 20 * sr), vec![0, 5 * sr]);
    }

    #[test]
    fn activity_marks_frames() {
        let segments = [seg(0.16, 0.40, 0), seg(0.0, 1.0, 1)];
        let mask = activity(&segments, 0, 0, SAMPLE_RATE);
        assert_eq!(mask.len(), 13);
        assert_eq!(mask, [0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]);
        // Relative to the chunk start.
        let mask = activity(&segments, 0, SAMPLE_RATE / 4, SAMPLE_RATE);
        assert_eq!(&mask[..3], &[1., 1., 0.]);
    }

    #[test]
    fn words_group_into_utterances() {
        let words = [
            word("Hello", 0.0, 0.4),
            word("there", 0.5, 0.9),
            word(".", 0.9, 0.9),
            word("Later", 3.0, 3.4),
        ];
        let out = group(2, words, 1.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "Hello there.");
        assert_eq!((out[0].start, out[0].end), (0.0, 0.9));
        assert_eq!(out[1].text, "Later");
        assert!(out.iter().all(|u| u.speaker_id == 2));
    }
}
