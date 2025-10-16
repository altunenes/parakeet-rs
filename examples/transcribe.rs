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
    println!("==========================================\n");
    println!("Loading Parakeet model from current directory...");
    let mut parakeet = Parakeet::from_pretrained(".")?;
    println!("Model loaded successfully!\n");

    println!("Transcribing audio file: {audio_path}");
    println!("Please wait, this may take a moment...\n");

    let result = parakeet.transcribe(audio_path)?;

    // just get the transcribed text
    println!("Transcription Result:");
    println!("---------------------");
    println!("{}", result.text);

    //  access token-level timestamps
    println!("\n\nToken-level Timestamps (first 10):");
    println!("-----------------------------------");
    for token in result.tokens.iter().take(10) {
        println!("[{:.3}s - {:.3}s] \"{}\"", token.start, token.end, token.text);
    }

    if result.tokens.len() > 10 {
        println!("... ({} more tokens)", result.tokens.len() - 10);
    }

    println!("\nTranscription completed successfully!");

    Ok(())
}
