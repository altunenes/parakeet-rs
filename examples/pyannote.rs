/*
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

CTC (English-only):
cargo run --example pyannote 6_speakers.wav

TDT (Multilingual):
cargo run --example pyannote 6_speakers.wav tdt

NOTE: If you do't want to depend on hound for WavSpec, see examples/raw.rs
- transcribe_raw() - without WavSpec dependency


*/

use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};
use std::env;
use std::time::Instant;
use hound;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    let audio_path = args.get(1)
        .expect("Please specify audio file: cargo run --example pyannote <audio.wav> [tdt]");

    let use_tdt = args.get(2).map(|s| s.as_str()) == Some("tdt");

    let max_speakers = 6;
    let speaker_threshold = 0.5;

   
    let (samples, sample_rate) = pyannote_rs::read_wav(audio_path)?;

    let mut extractor = EmbeddingExtractor::new("wespeaker_en_voxceleb_CAM++.onnx")?;
    let mut manager = EmbeddingManager::new(max_speakers);

    let segments: Vec<_> =
        pyannote_rs::get_segments(&samples, sample_rate, "segmentation-3.0.onnx")?.collect();

    // Build speaker map: segment_index -> speaker_label
    let mut segment_speakers = Vec::new();
    for segment_result in segments {
        if let Ok(segment) = segment_result {
            let duration = segment.end - segment.start;
            if duration < 0.5 {
                continue;
            }

            let speaker = if let Ok(embedding) = extractor.compute(&segment.samples) {
                if manager.get_all_speakers().len() == max_speakers {
                    manager
                        .get_best_speaker_match(embedding.collect())
                        .map(|s| s.to_string())
                        .unwrap_or("UNKNOWN".to_string())
                } else {
                    manager
                        .search_speaker(embedding.collect(), speaker_threshold)
                        .map(|s| s.to_string())
                        .unwrap_or("UNKNOWN".to_string())
                }
            } else {
                "UNKNOWN".to_string()
            };

            segment_speakers.push((segment.start, segment.end, speaker));
        }
    }

    //Transcribe each speaker segment
    println!("{}", "=".repeat(80));

    if use_tdt {
        // TDT: Transcribe each speaker segment
        let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained("./tdt", None)?;

        // TDT needs more context than CTC - add padding before/after segments
        let padding_before = (0.2 * sample_rate as f64) as usize;
        let padding_after = (0.1 * sample_rate as f64) as usize;

        for (seg_start, seg_end, speaker) in &segment_speakers {
            // Extract segment audio samples with padding
            let start_sample = ((*seg_start * sample_rate as f64) as usize)
                .saturating_sub(padding_before);
            let end_sample = (((*seg_end * sample_rate as f64) as usize) + padding_after)
                .min(samples.len());

            if start_sample >= samples.len() || end_sample > samples.len() {
                continue;
            }

            let segment_samples = &samples[start_sample..end_sample];

            // Write segment to temporary file
            let temp_path = format!("/tmp/segment_{}_{}.wav", seg_start, seg_end);
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: sample_rate as u32,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&temp_path, spec)?;
            for &sample in segment_samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;

            // Transcribe segment
            if let Ok(result) = parakeet.transcribe_file(&temp_path) {
                if !result.text.trim().is_empty() {
                    println!("\n[{:.2}s - {:.2}s] Speaker {}:", seg_start, seg_end, speaker);
                    println!("  {}", result.text.trim());
                }
            }

            // Clean up temp file
            let _ = std::fs::remove_file(&temp_path);
        }
    } else {
        // CTC: Transcribe each speaker segment individually
        let mut parakeet = parakeet_rs::Parakeet::from_pretrained(".", None)?;

        for (seg_start, seg_end, speaker) in &segment_speakers {
            // Extract segment audio samples
            let start_sample = (*seg_start * sample_rate as f64) as usize;
            let end_sample = (*seg_end * sample_rate as f64) as usize;

            if start_sample >= samples.len() || end_sample > samples.len() {
                continue;
            }

            let segment_samples = &samples[start_sample..end_sample];

            // Write segment to temporary file
            let temp_path = format!("/tmp/segment_{}_{}.wav", seg_start, seg_end);
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: sample_rate as u32,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&temp_path, spec)?;
            for &sample in segment_samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;

            // Transcribe segment
            if let Ok(result) = parakeet.transcribe_file(&temp_path) {
                if !result.text.trim().is_empty() {
                    println!("\n[{:.2}s - {:.2}s] Speaker {}:", seg_start, seg_end, speaker);
                    println!("  {}", result.text.trim());
                }
            }

            // Clean up temp file
            let _ = std::fs::remove_file(&temp_path);
        }
    }

    println!("\n{}", "=".repeat(80));
    let elapsed = start_time.elapsed();
    println!("\n✓ Transcription completed in {:.2}s", elapsed.as_secs_f32());

    Ok(())
}
