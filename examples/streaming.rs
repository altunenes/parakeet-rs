/*
Demonstrates streaming ASR with Parakeet RealTime EOU

Download models files from:
https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx

This example
- Maintains 4-second ring buffer for feature extraction context
- Processes 160ms chunks (2560 samples at 16kHz)
- Extracts features from full buffer, then slices last 25 frames
- Encoder receives: 9 frames (pre-encode cache) + 16 frames (new) = 25 total
- Cache states (cache_last_channel/time) maintain temporal context

Model files required in ./fullstr/:
  - encoder.onnx (cache_aware_stream_step export)
  - decoder_joint.onnx
  - tokenizer.json

Additional notes:
let reset_on_eou: bool = false;
I must admit that this is not work very well on my real world tests :/


Usage:
cargo run --release --example streaming <audio.wav>
*/

use parakeet_rs::ParakeetEOU;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let args: Vec<String> = env::args().collect();
    let audio_path = args
        .get(1)
        .expect("Usage: cargo run --release --example streaming <audio.wav>");

    println!("Loading model from ./fullstr...");
    let mut parakeet = ParakeetEOU::from_pretrained("./fullstr", None)?;

    println!("Loading audio: {}", audio_path);
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    if spec.sample_rate != 16000 {
        return Err(format!(
            "Expected 16kHz audio, got {}Hz. Please resample first.",
            spec.sample_rate
        )
        .into());
    }

    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    let max_val = audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_val > 1e-6 {
        let norm_factor = max_val + 1e-5;
        for sample in &mut audio {
            *sample /= norm_factor;
        }
    }

    let duration = audio.len() as f32 / 16000.0;
    // 160ms at 16kHz
    const CHUNK_SIZE: usize = 2560;
    let reset_on_eou: bool = false;

    println!("Streaming transcription (160ms chunks with 4s buffer)...\n");

    let mut full_text = String::new();

    for chunk in audio.chunks(CHUNK_SIZE) {
        let chunk_vec = if chunk.len() < CHUNK_SIZE {
            let mut padded = chunk.to_vec();
            padded.resize(CHUNK_SIZE, 0.0);
            padded
        } else {
            chunk.to_vec()
        };

        let text = parakeet.transcribe(&chunk_vec, reset_on_eou)?;
        if !text.is_empty() {
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout())?;
            full_text.push_str(&text);
        }
    }

    println!("\n\nFlushing decoder...");
    let silence = vec![0.0f32; CHUNK_SIZE];
    for _ in 0..3 {
        let text = parakeet.transcribe(&silence, reset_on_eou)?;
        if !text.is_empty() {
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout())?;
            full_text.push_str(&text);
        }
    }

    println!("\n\nFinal Transcription:\n{}", full_text.trim());

    let elapsed = start_time.elapsed();
    println!(
        "\nTranscription completed in {:.2}s (audio: {:.2}s, RTF: {:.2}x)",
        elapsed.as_secs_f32(),
        duration,
        duration / elapsed.as_secs_f32()
    );

    Ok(())
}
