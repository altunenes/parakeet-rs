/*
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

CTC (English-only):
cargo run --example pyannote 6_speakers.wav

TDT (Multilingual):
cargo run --example pyannote 6_speakers.wav tdt

NOTE: This example demonstrates pyannote for speaker diarization (speaker identification),
but uses Parakeet's Sentences-level timestamps. you can directly use pyannote's timestamps also.
Pyannote's rust version is still experimental, please check this for further discussions:
https://github.com/thewh1teagle/pyannote-rs/pull/24

*/

use parakeet_rs::TimestampMode;
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

    // Transcribe entire audio
    println!("{}", "=".repeat(80));
    println!("\nSentencess:");

    if use_tdt {
        let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained("./tdt", None)?;

        // Create temp file for full audio, you can also use directly samplees so that this step is not needed
        let temp_path = "/tmp/full_audio.wav";
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(temp_path, spec)?;
        for &sample in &samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;

        // Transcribe with Sentences
        if let Ok(result) = parakeet.transcribe_file(temp_path, Some(TimestampMode::Sentences)) {
            // For each transcription segment, find the corresponding speaker
            for segment in &result.tokens {
                // Find which speaker was active during this segment's midpoint
                let segment_mid = ((segment.start + segment.end) / 2.0) as f64;
                let speaker = segment_speakers
                    .iter()
                    .find(|(start, end, _)| segment_mid >= *start && segment_mid <= *end)
                    .map(|(_, _, s)| s.clone())
                    .unwrap_or_else(|| "UNKNOWN".to_string());

                println!("[{:.2}s - {:.2}s] Speaker {}: {}",
                    segment.start, segment.end, speaker, segment.text);
            }
        }

        let _ = std::fs::remove_file(temp_path);
    } else {
        let mut parakeet = parakeet_rs::Parakeet::from_pretrained(".", None)?;

        // Create temp file for full audio
        let temp_path = "/tmp/full_audio.wav";
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(temp_path, spec)?;
        for &sample in &samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;

        // Transcribe with Sentences granularity - this gives better timestamps!
        if let Ok(result) = parakeet.transcribe_file(temp_path, Some(TimestampMode::Sentences)) {
            // For each transcription segment, find the corresponding speaker
            for segment in &result.tokens {
                // Find which speaker was active during this segment's midpoint
                let segment_mid = ((segment.start + segment.end) / 2.0) as f64;
                let speaker = segment_speakers
                    .iter()
                    .find(|(start, end, _)| segment_mid >= *start && segment_mid <= *end)
                    .map(|(_, _, s)| s.clone())
                    .unwrap_or_else(|| "UNKNOWN".to_string());

                println!("[{:.2}s - {:.2}s] Speaker {}: {}",
                    segment.start, segment.end, speaker, segment.text);
            }
        }

        let _ = std::fs::remove_file(temp_path);
    }

    println!("\n{}", "=".repeat(80));
    let elapsed = start_time.elapsed();
    println!("\n✓ Transcription completed in {:.2}s", elapsed.as_secs_f32());

    Ok(())
}
