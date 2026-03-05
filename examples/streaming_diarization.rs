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

Streaming config (chunk_len/fifo_len/spkcache_len):
- Read automatically from ONNX model metadata if present (default: 124/124/188)
- Query latency after construction: sortformer.latency() returns chunk duration in seconds
- Override for different latency: sf.chunk_len = 62; sf.fifo_len = 62; sf.spkcache_len = 94;
- Smaller chunks = lower latency but reduced accuracy
- The ONNX graph uses dynamic axes, so chunk sizes work at runtime
- NOTE: Defaults (124/124/188) match NVIDIA's training config and give best accuracy
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
use std::time::{Duration, Instant};

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
            "diar_streaming_sortformer_4spk-v2.1.onnx",
            None,
            DiarizationConfig::callhome(),
        )?;

        let chunk_size = sortformer.chunk_len * 80 * 16;
        let feed_size = sortformer.right_context * 80 * 16 + chunk_size;
        let mut counter = 0;
        loop {
            let chunk_start = counter * chunk_size;

            if chunk_start >= audio.len() {
                break;
            }

            let chunk_end = (chunk_start + feed_size).min(audio.len());

            let feed_data = &audio[chunk_start..chunk_end];

            let speaker_segments = sortformer.diarize_chunk(feed_data)?;

            let chunk_ts = chunk_start as f32 / 16_000.;
            let chunk_duration = chunk_size.min(feed_data.len()) as f32 / 16_000.;

            // Print raw diarization segments
            for seg in &speaker_segments {
                if seg.start > chunk_duration {
                    break;
                }

                let start = Duration::from_secs_f32(seg.start + chunk_ts);

                let end = if seg.end > chunk_duration {
                    Duration::from_secs_f32(chunk_ts + chunk_duration)
                } else {
                    Duration::from_secs_f32(chunk_ts + seg.end)
                };

                println!(
                    "  [{:0>2}:{:0>2}:{:0>2}.{:0>3} - {:0>2}:{:0>2}:{:0>2}.{:0>3}] Speaker {}",
                    start.as_secs() / 3_600,
                    (start.as_secs() / 60) % 60,
                    start.as_secs() % 60,
                    start.subsec_millis(),
                    end.as_secs() / 3_600,
                    (end.as_secs() / 60) % 60,
                    end.as_secs() % 60,
                    end.subsec_millis(),
                    seg.speaker_id
                );
            }

            counter += 1;
        }

        /*
        for chunk in audio.chunks(chunk_size) {
            counter += 1;
            let speaker_segments =
                sortformer.diarize_chunk(chunk.clone())?;

            println!(
                "Found {} speaker segments from Sortformer after feeding {} seconds",
                speaker_segments.len(), (counter * chunk_size) as f64 / 16_000.,
            );

            // Print raw diarization segments
            println!("\nRaw diarization segments:");
            for seg in &speaker_segments {
                println!(
                    "  [{:06.2}s - {:06.2}s] Speaker {}",
                    seg.start, seg.end, seg.speaker_id
                );
            }
            break;
        }
        */

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
