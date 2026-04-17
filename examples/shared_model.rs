/// Shared NemotronModel — API demo.
///
/// Load the model once via [`NemotronHandle`], create two instances with
/// independent decoder state, and feed the same audio to both to confirm
/// deterministic output.
///
/// Usage:
///   cargo run --release --example shared_model <model_dir> <audio.wav>

use parakeet_rs::Nemotron;

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    Ok(audio)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: shared_model <model_dir> <audio.wav>");
        std::process::exit(1);
    }

    // Load the ONNX model once (~1.4 GB) into an opaque handle.
    let handle = Nemotron::load_model(&args[1], None)?;

    // Create two instances sharing the model, each with independent decoder state.
    let mut a = Nemotron::from_shared_model(handle.clone());
    let mut b = Nemotron::from_shared_model(handle);

    let audio = load_wav(&args[2])?;
    let chunk_size = 8960; // 560 ms at 16 kHz

    for chunk_data in audio.chunks(chunk_size) {
        let mut chunk = chunk_data.to_vec();
        chunk.resize(chunk_size, 0.0);
        a.transcribe_chunk(&chunk)?;
        b.transcribe_chunk(&chunk)?;
    }

    println!("A: {}", a.get_transcript());
    println!("B: {}", b.get_transcript());
    assert_eq!(a.get_transcript(), b.get_transcript(), "shared model must be deterministic");
    println!("OK — both instances produced identical output");
    Ok(())
}
