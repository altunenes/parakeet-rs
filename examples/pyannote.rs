/*
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
cargo run --example pyannote 6_speakers.wav
*/

use parakeet_rs::Parakeet;
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = env::args()
        .nth(1)
        .expect("Please specify audio file: cargo run --example pyannote <audio.wav>");

    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path)?;

    let max_speakers = 6;
    let speaker_threshold = 0.5;

    // Load model from current directory (auto-detects with priority: model.onnx > model_fp16.onnx > model_int8.onnx > model_q4.onnx)
    // Or specify exact model: Parakeet::from_pretrained("model_q4.onnx", None)?
    let mut parakeet = Parakeet::from_pretrained(".", None)?;

    let mut extractor = EmbeddingExtractor::new("wespeaker_en_voxceleb_CAM++.onnx")?;
    let mut manager = EmbeddingManager::new(max_speakers);

    let segments: Vec<_> =
        pyannote_rs::get_segments(&samples, sample_rate, "segmentation-3.0.onnx")?.collect();

    println!("{}", "=".repeat(80));

    // Process each segment
    for (idx, segment_result) in segments.into_iter().enumerate() {
        if let Ok(segment) = segment_result {
            let duration = segment.end - segment.start;

            // Skip very short segments (< 0.5s)
            if duration < 0.5 {
                continue;
            }

            // Check if segment is too long (> 30s recommended for ASR models)
            let max_duration_secs = 30.0;
            if duration > max_duration_secs {
                eprintln!("Warning: Segment {idx} too long ({duration:.1}s), skipping");
                continue;
            }

            // Identify speaker
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

            // Save segment to temporary WAV file
            let temp_path = format!("/tmp/segment_{idx}.wav");
            if let Err(e) = save_segment_as_wav(&temp_path, &segment.samples, sample_rate) {
                eprintln!("Warning: Failed to save segment {idx}: {e}");
                continue;
            }

            // Transcribe
            let result = match parakeet.transcribe(&temp_path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Warning: Failed to transcribe segment {idx}: {e}");
                    let _ = std::fs::remove_file(&temp_path);
                    continue;
                }
            };

            let _ = std::fs::remove_file(&temp_path);

            // Only print if we got actual text
            let text = result.text.trim();
            if !text.is_empty() {
                println!(
                    "\n[{:.2}s - {:.2}s] Speaker {speaker}:",
                    segment.start, segment.end
                );
                println!("  {text}");
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("\n✓ Transcription completed!");

    Ok(())
}

/// Save audio segment as WAV file, resampling to 16kHz if needed
fn save_segment_as_wav(
    path: &str,
    samples: &[i16],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use hound::{WavSpec, WavWriter};

    let target_rate = 16000;

    // Check for reasonable segment length (30s max)
    let max_samples = 16000 * 30;
    if samples.len() > max_samples * 2 {
        return Err("Segment too long".into());
    }

    // Resample if needed
    let output_samples: Vec<i16> = if sample_rate != target_rate {
        let ratio = sample_rate as f64 / target_rate as f64;
        let output_len = (samples.len() as f64 / ratio) as usize;

        if output_len > max_samples {
            return Err("Resampled segment too long".into());
        }

        (0..output_len)
            .map(|i| {
                let src_idx = (i as f64 * ratio) as usize;
                samples.get(src_idx).copied().unwrap_or(0)
            })
            .collect()
    } else {
        if samples.len() > max_samples {
            return Err("Segment too long".into());
        }
        samples.to_vec()
    };

    let spec = WavSpec {
        channels: 1,
        sample_rate: target_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in &output_samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}
