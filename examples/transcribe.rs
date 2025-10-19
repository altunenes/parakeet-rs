/*
transcribes entire audio, no diarization
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
cargo run --example transcribe 6_speakers.wav
*/
use parakeet_rs::Parakeet;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "6_speakers.wav"
    };
    // Load model from current directory (auto-detects with priority: model.onnx > model_fp16.onnx > model_int8.onnx > model_q4.onnx)
    // Or specify exact model: Parakeet::from_pretrained("model_q4.onnx")
    let mut parakeet = Parakeet::from_pretrained(".")?;

    let result = parakeet.transcribe(audio_path)?;

    // Print transcription
    println!("{}", result.text);

    // Access token-level timestamps
    println!("\nFirst 10 tokens:");
    for token in result.tokens.iter().take(10) {
        println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
    }

    Ok(())
}
