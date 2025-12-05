/*
Demonstrates using transcribe_samples()

This example shows manual audio loading and calling transcribe_samples() directly
with sample_rate and channels instead of using transcribe_file()

Usage (default: ctc):
cargo run --example raw 6_speakers.wav
cargo run --example raw 6_speakers.wav tdt
cargo run --example raw 6_speakers.wav canary

WARNING: TDT model has sequence length limitations (~8-10 minutes max).
For longer audio files, you must split into chunks (e.g., 5-minute segments)
and transcribe each chunk separately. Attempting to transcribe 25+ minute
audio files in one call will cause ONNX runtime errors.
Otherwise you will likely get a error like:
"Error: Ort(Error { code: RuntimeException, msg: "Non-zero status code returned while running Add node. Name:'/layers.0/self_attn/Add_2' Status Message: /Users/runner/work/ort-artifacts/ort-artifacts/onnxruntime/onnxruntime/core/providers/cpu/math/element_wise_ops.h:540 void onnxruntime::BroadcastIterator::Init(ptrdiff_t, ptrdiff_t) axis == 1 || axis == largest was false. })"
*/

use parakeet_rs::{Parakeet, ParakeetCanary, ParakeetTDT, TimestampMode, Transcriber};
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

    let model_type = args.get(2).map(|s| s.as_str()).unwrap_or("ctc");

    // Load audio manually using hound (or any other audio library)
    // remember if you use raw audio API, you need to handle audio preprocessing yourself!
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    println!(
        "Audio info: {}Hz, {} channel(s)",
        spec.sample_rate, spec.channels
    );

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    match model_type {
        "canary" => {
            println!("Loading Canary model...");
            let mut canary = ParakeetCanary::from_pretrained("./canary", None)?;

            // Set source and target languages (both default to "en")
            canary.set_language("en");

            // Translation: source ≠ target (e.g., Swedish audio → English text)
            // canary.set_source_language("en");
            // canary.set_target_language("en");

            let result = canary.transcribe_samples(
                audio,
                spec.sample_rate,
                spec.channels,
                Some(TimestampMode::Sentences),
            )?;

            println!("{}", result.text);

            if !result.tokens.is_empty() {
                println!("\nSentences:");
                for token in result.tokens.iter() {
                    println!("[{:.2}s - {:.2}s]: {}", token.start, token.end, token.text);
                }
            }
        }
        "tdt" => {
            println!("Loading TDT model...");
            let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;

            // Use transcribe_samples() with raw parameters and timestamp mode
            let result = parakeet.transcribe_samples(
                audio,
                spec.sample_rate,
                spec.channels,
                Some(TimestampMode::Sentences),
            )?;

            println!("{}", result.text);
            println!("\nSentences:");
            for segment in result.tokens.iter() {
                println!(
                    "[{:.2}s - {:.2}s]: {}",
                    segment.start, segment.end, segment.text
                );
            }
        }
        _ => {
            println!("Loading CTC model...");
            let mut parakeet = Parakeet::from_pretrained(".", None)?;

            // CTC model doesn't predict punctuation (lowercase alphabet only)
            // This means no sentence boundaries. we use Words mode instead of Sentences
            let result = parakeet.transcribe_samples(
                audio,
                spec.sample_rate,
                spec.channels,
                Some(TimestampMode::Words),
            )?;

            println!("{}", result.text);

            // Access word-level timestamps (showing first 10 for brevity)
            // Note: CTC generates word-level timestamps but cannot segment into sentences
            // due to lack of punctuation prediction - this is a model limitation if I not mistake
            println!("\nWords (first 10):");
            for word in result.tokens.iter().take(10) {
                println!("[{:.2}s - {:.2}s]: {}", word.start, word.end, word.text);
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "\n✓ Transcription completed in {:.2}s",
        elapsed.as_secs_f32()
    );

    Ok(())
}
