/*
Streaming Speaker Diarization with NVIDIA Nemotron-3 Diarization (Sortformer v3).

Download nemotron3_diar_v3.onnx from https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3-diarization
(or export it with scripts/export_diar_sortformer.py), then feed audio in small
chunks as it arrives (mic, GStreamer, etc.). State (speaker cache, FIFO) is preserved
across feed() calls, so speaker IDs stay consistent over time.

Latency vs accuracy (same ONNX, no re-export; set before feeding audio, it resets state):
  Buffer latency = (chunk_len + right_context) * 80ms. NVIDIA's recommended presets:
    - StreamingProfile::offline()            30.4 s
    - StreamingProfile::low_latency()        1.04 s   <- used here
    - StreamingProfile::very_low_latency()   0.64 s
    - StreamingProfile::ultra_low_latency()  0.32 s
  Lower latency means less context per step and slightly higher DER (see the model card).

Resolution: segments come from the model's 10ms "hires" frames (80ms frames are used
internally for the speaker cache), so timestamps are accurate to ~10ms. Note feed()
binarizes each chunk independently, so a segment crossing a chunk edge may be split --
per-speaker total voiced time still matches the whole-file result.

Usage:
  cargo run --release --example streaming-diarization --features sortformer -- <audio.wav> [nemotron3_diar_v3.onnx]
*/

#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{Sortformer, StreamingProfile};
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!("Run with: cargo run --example streaming-diarization --features sortformer -- <audio.wav>");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args
            .get(1)
            .expect("usage: streaming-diarization --features sortformer -- <audio.wav> [onnx]");
        let onnx = args.get(2).map(String::as_str).unwrap_or("nemotron3_diar_v3.onnx");

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
        println!("Loaded {:.1}s of audio", audio.len() as f32 / 16_000.0);

        let mut diarizer = Sortformer::new(onnx)?;
        diarizer.set_profile(StreamingProfile::low_latency())?;
        let p = diarizer.profile();
        println!(
            "Profile: chunk_len={}, right_context={}, latency={:.2}s",
            p.chunk_len,
            p.right_context,
            diarizer.latency()
        );

        // Simulate real-time streaming: feed 20ms chunks (in practice, from a mic/GStreamer).
        let feed_chunk_size = 320; // 20ms at 16kHz
        let mut total_segments = 0;
        println!("\nStreaming diarization (feeding {}ms chunks):", feed_chunk_size * 1000 / 16_000);
        println!("{}", "-".repeat(60));

        for chunk in audio.chunks(feed_chunk_size) {
            for seg in diarizer.feed(chunk)? {
                println!(
                    "  [{:06.2}s - {:06.2}s] speaker_{}",
                    seg.start as f64 / 16_000.0,
                    seg.end as f64 / 16_000.0,
                    seg.speaker_id
                );
                total_segments += 1;
            }
        }
        for seg in diarizer.flush()? {
            println!(
                "  [{:06.2}s - {:06.2}s] speaker_{} (flush)",
                seg.start as f64 / 16_000.0,
                seg.end as f64 / 16_000.0,
                seg.speaker_id
            );
            total_segments += 1;
        }

        println!("{}", "-".repeat(60));
        println!("Done: {} segments in {:.2}s", total_segments, start_time.elapsed().as_secs_f32());
        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
