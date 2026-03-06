/*
Speaker Diarization with NVIDIA Sortformer v2 (Streaming)

Download the Sortformer v2 model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx
Or download the Sortformer v2.1 model:
https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.1.onnx
Download test audio:
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

Usage:
cargo run --example streaming-diarization --features sortformer <audio.wav>
*/

#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!("Run with: cargo run --example streaming_diarization --features sortformer <audio.wav>");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args.get(1).expect(
            "Please specify audio file: cargo run --example streaming-diarization --features sortformer <audio.wav>",
        );

        // Load audio
        let mut reader = hound::WavReader::open(audio_path)?;
        let spec = reader.spec();

        if spec.sample_rate != 16000 {
            return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
        }

        let mut audio: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?,
        };

        if spec.channels > 1 {
            audio = audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect();
        }

        let duration = audio.len() as f32 / 16_000.0;
        println!("Loaded {:.1}s of audio", duration);

        // Create Sortformer
        let mut sortformer = Sortformer::with_config(
            "diar_streaming_sortformer_4spk-v2.1.onnx",
            None,
            DiarizationConfig::callhome(),
        )?;

        println!(
            "Config: chunk_len={}, right_context={}, latency={:.2}s",
            sortformer.chunk_len,
            sortformer.right_context,
            sortformer.latency()
        );

        // simulate real-time streaming: feed small chunks (for instance 20ms = 320 samples)
        // In practice, real world these would come from gsttreamer, mic etc ofc
        let feed_chunk_size = 320; // 20ms at 16kHz
        let mut total_segments = 0;

        println!("\nStreaming diarization (feeding {}ms chunks):", feed_chunk_size * 1000 / 16_000);
        println!("{}", "-".repeat(60));

        for chunk in audio.chunks(feed_chunk_size) {
            let segments = sortformer.feed(chunk)?;

            for seg in &segments {
                println!(
                    "  [{:06.2}s - {:06.2}s] Speaker {}",
                    seg.start as f64 / 16_000.0,
                    seg.end as f64 / 16_000.0,
                    seg.speaker_id
                );
            }
            total_segments += segments.len();
        }

        // Flush remaining buffered audio
        let final_segments = sortformer.flush()?;
        for seg in &final_segments {
            println!(
                "  [{:06.2}s - {:06.2}s] Speaker {} (flush)",
                seg.start as f64 / 16_000.0,
                seg.end as f64 / 16_000.0,
                seg.speaker_id
            );
        }
        total_segments += final_segments.len();

        println!("{}", "-".repeat(60));
        println!(
            "Done: {} segments in {:.2}s",
            total_segments,
            start_time.elapsed().as_secs_f32()
        );

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
