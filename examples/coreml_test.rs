/*
Minimal reproduction for CoreML execution provider issue.
This example explicitly forces CoreML and will fail with the error.

cargo run --example coreml_test --features coreml --release
*/
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Parakeet};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Explicitly use CoreML execution provider (no CPU fallback)
    let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::CoreML);
    let mut parakeet = Parakeet::from_pretrained_with_config(".", config)?;
    let result = parakeet.transcribe("6_speakers.wav")?;
    println!("Transcription: {}", result.text);
    Ok(())
}
