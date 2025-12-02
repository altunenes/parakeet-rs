/*
Speaker Diarization with NVIDIA Sortformer v2 (Streaming)

Download the Sortformer v2 model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx
Or download the Sortformer v2.1 model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.1.onnx
Download test audio:
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

Usage:
cargo run --example diarization --features sortformer 6_speakers.wav

NOTE: This example combines two NVIDIA models:
- Parakeet-TDT: Provides transcription with sentence-level timestamps
- Sortformer v2: Provides streaming speaker identification (4 speakers max)
- We use TDT's sentence timestamps + Sortformer's speaker IDs
- Even if Sortformer can't detect a segment, we still get the transcription (marked UNKNOWN)
- For more information:
https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2

WARNING: Sortformer handles long audio natively (streaming), but TDT has sequence
length limitations (~8-10 minutes max). For production use with long audio files,
run Sortformer on the full audio for diarization, then chunk the audio into
~5-minute segments for TDT transcription, and map the results back together.
*/

#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use parakeet_rs::{TimestampMode, Transcriber};
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!(
            "Please run with: cargo run --example diarization --features sortformer <audio.wav>"
        );
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args.get(1)
            .expect("Please specify audio file: cargo run --example diarization --features sortformer <audio.wav>");

        println!("{}", "=".repeat(80));
        println!("Step 1/3: Loading audio...");

        let mut reader = hound::WavReader::open(audio_path)?;
        let spec = reader.spec();

        let audio: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?,
        };

        let duration = audio.len() as f32 / spec.sample_rate as f32 / spec.channels as f32;
        println!(
            "Loaded {} samples ({} Hz, {} channels, {:.1}s)",
            audio.len(),
            spec.sample_rate,
            spec.channels,
            duration
        );

        println!("{}", "=".repeat(80));
        println!("Step 2/3: Performing speaker diarization with Sortformer v2 (streaming)...");

        // Create Sortformer with default config (callhome)
        let mut sortformer = Sortformer::with_config(
            "diar_streaming_sortformer_4spk-v2.onnx",
            None, // default exec config
            DiarizationConfig::callhome(),
        )?;

        let speaker_segments =
            sortformer.diarize(audio.clone(), spec.sample_rate, spec.channels)?;

        println!(
            "Found {} speaker segments from Sortformer",
            speaker_segments.len()
        );

        // Print raw diarization segments
        println!("\nRaw diarization segments:");
        for seg in &speaker_segments {
            println!(
                "  [{:06.2}s - {:06.2}s] Speaker {}",
                seg.start, seg.end, seg.speaker_id
            );
        }

        println!("\n{}", "=".repeat(80));
        println!("Step 3/3: Transcribing with Parakeet-TDT and attributing speakers...\n");

        // Use TDT for transcription with sentence-level timestamps
        let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained("./tdt", None)?;

        // Transcribe with Sentences mode (TDT provides punctuation for proper segmentation)
        if let Ok(result) = parakeet.transcribe_samples(
            audio,
            spec.sample_rate,
            spec.channels,
            Some(TimestampMode::Sentences),
        ) {
            // For each sentence from TDT, find the corresponding speaker from Sortformer
            for segment in &result.tokens {
                // Find speaker with maximum overlap
                let speaker = speaker_segments
                    .iter()
                    .filter_map(|s| {
                        // Calculate overlap between transcription and diarization segment
                        let overlap_start = segment.start.max(s.start);
                        let overlap_end = segment.end.min(s.end);
                        let overlap = (overlap_end - overlap_start).max(0.0);
                        if overlap > 0.0 {
                            Some((s.speaker_id, overlap))
                        } else {
                            None
                        }
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(id, _)| format!("Speaker {}", id))
                    .unwrap_or_else(|| "UNKNOWN".to_string());

                println!(
                    "[{:.2}s - {:.2}s] {}: {}",
                    segment.start, segment.end, speaker, segment.text
                );
            }
        }

        println!("\n{}", "=".repeat(80));
        let elapsed = start_time.elapsed();
        println!(
            "\n✓ Diarization and transcription completed in {:.2}s",
            elapsed.as_secs_f32()
        );
        println!("• UNKNOWN: Segments where no speaker was detected by Sortformer");
        println!("• Config: callhome v2 (onset=0.641, offset=0.561, min_on=0.511, min_off=0.296)");

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
