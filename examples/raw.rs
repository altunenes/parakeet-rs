/*
Demonstrates using raw audio API (No WavSpec dependency!!)

This example shows using transcribe_raw() instead of transcribe_file()

Usage:
cargo run --example raw 6_speakers.wav
cargo run --example raw 6_speakers.wav tdt
*/

use parakeet_rs::{Parakeet, ParakeetTDT};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "6_speakers.wav"
    };

    let use_tdt = args.len() > 2 && args[2] == "tdt";

    // Load audio manually using hound (or any other audio library)
    // remember if you use raw audio API, you need to handle audio preprocessing yourself!
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    println!("Audio info: {}Hz, {} channel(s)", spec.sample_rate, spec.channels);

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    if use_tdt {
        println!("Loading TDT model...");
        let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;

        // Use transcribe_raw() -  without WavSpec
        let result = parakeet.transcribe_raw(audio, spec.sample_rate, spec.channels)?;

        println!("{}", result.text);
        println!("\nFirst 10 tokens:");
        for token in result.tokens.iter().take(10) {
            println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
        }
    } else {
        // CTC also supports transcribe_raw()
        println!("Loading CTC model...");
        let mut parakeet = Parakeet::from_pretrained(".", None)?;

        let result = parakeet.transcribe_raw(audio, spec.sample_rate, spec.channels)?;

        println!("{}", result.text);
        println!("\nFirst 10 tokens:");
        for token in result.tokens.iter().take(10) {
            println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\n✓ Transcription completed in {:.2}s", elapsed.as_secs_f32());

    Ok(())
}
