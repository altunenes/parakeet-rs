/*
transcribes entire audio, no diarization
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav

CTC (English-only):
cargo run --example transcribe 6_speakers.wav

TDT (Multilingual):
cargo run --example transcribe 6_speakers.wav tdt

WARNING: This may fail on very long audio files (>8 min).
For longer audio, use the pyannote example which processes segments, or split your audio into chunks.

Note: The coreml feature flag is only for reproducing a known ONNX Runtime bug.
Just ignore it :). See: https://github.com/microsoft/onnxruntime/issues/26355
*/
use parakeet_rs::Parakeet;
use std::env;
use std::time::Instant;

#[cfg(feature = "coreml")]
use parakeet_rs::{ExecutionConfig, ExecutionProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "6_speakers.wav"
    };

    let use_tdt = args.len() > 2 && args[2] == "tdt";

    // TDT model (multilingual, 25 languages)
    if use_tdt {
        #[cfg(feature = "coreml")]
        {
            let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::CoreML);
            let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained("./tdt", Some(config))?;
            let result = parakeet.transcribe_file(audio_path)?;
            println!("{}", result.text);

            println!("\nFirst 10 tokens:");
            for token in result.tokens.iter().take(10) {
                println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
            }

            let elapsed = start_time.elapsed();
            println!("\n✓ Transcription completed in {:.2}s", elapsed.as_secs_f32());
            return Ok(());
        }

        #[cfg(not(feature = "coreml"))]
        {
            let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained("./tdt", None)?;
            let result = parakeet.transcribe_file(audio_path)?;
            println!("{}", result.text);

            println!("\nFirst 10 tokens:");
            for token in result.tokens.iter().take(10) {
                println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
            }

            let elapsed = start_time.elapsed();
            println!("\n✓ Transcription completed in {:.2}s", elapsed.as_secs_f32());
            return Ok(());
        }
    }

    // CTC model (English-only)
    #[cfg(feature = "coreml")]
    let mut parakeet = {
        let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::CoreML);
        Parakeet::from_pretrained(".", Some(config))?
    };

    // Default: CPU execution provider (works correctly)
    // Auto-detects model with priority: model.onnx > model_fp16.onnx > model_int8.onnx > model_q4.onnx
    // Or specify exact model: Parakeet::from_pretrained("model_q4.onnx", None)?
    #[cfg(not(feature = "coreml"))]
    let mut parakeet = Parakeet::from_pretrained(".", None)?;

    let result = parakeet.transcribe_file(audio_path)?;

    // Print transcription
    println!("{}", result.text);

    // Access token-level timestamps
    println!("\nFirst 10 tokens:");
    for token in result.tokens.iter().take(10) {
        println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
    }

    let elapsed = start_time.elapsed();
    println!("\n✓ Transcription completed in {:.2}s", elapsed.as_secs_f32());

    Ok(())
}
