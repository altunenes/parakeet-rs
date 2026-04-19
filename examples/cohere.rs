/*
Cohere Transcribe ASR — offline multilingual transcription.

2B parameter encoder-decoder model supporting 14 languages with inverse text norm.

Download the ONNX export from:
https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX

Usage:
  cargo run --release --example cohere --features cohere -- \
    <model_dir> <audio.wav> [lang] [pnc] [itn]

Examples:
  cargo run --release --example cohere --features cohere -- ./cohere audio.wav
  cargo run --release --example cohere --features cohere -- ./cohere audio.wav en true false
  cargo run --release --example cohere --features cohere -- ./cohere audio.wav ja true true

Languages: ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh
*/

#[cfg(feature = "cohere")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::CohereASR;
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_dir> <audio.wav> [lang] [pnc] [itn]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let audio_path = &args[2];
    let language = args.get(3).map(String::as_str).unwrap_or("en");
    let pnc = args.get(4).map(|s| s == "true").unwrap_or(true);
    let itn = args.get(5).map(|s| s == "true").unwrap_or(false);

    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz audio, got {}Hz", spec.sample_rate).into());
    }

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    let duration_secs = audio.len() as f32 / 16000.0;
    println!(
        "Audio: {:.2}s, language={}, pnc={}, itn={}",
        duration_secs, language, pnc, itn
    );

    println!("Loading Cohere model...");
    let load_start = Instant::now();
    let mut model = CohereASR::from_pretrained(model_dir, None)?;
    println!("Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    let start = Instant::now();
    let text = model.transcribe_audio(&audio, language, pnc, itn)?;
    let elapsed = start.elapsed().as_secs_f32();

    println!("\n{}", text);
    println!(
        "\nTranscribed in {:.2}s (RTF {:.2}x)",
        elapsed,
        duration_secs / elapsed
    );

    Ok(())
}

#[cfg(not(feature = "cohere"))]
fn main() {
    eprintln!("Rebuild with --features cohere to run this example.");
    std::process::exit(1);
}
