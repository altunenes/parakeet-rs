/*
Shared-model demo — load one ONNX session, drive two concurrent streams
with independent decoder state for streaming models:
Nemotron (default):
cargo run --release --example shared_model ./nemotron audio.wav

EOU:
cargo run --release --example shared_model ./fullstr audio.wav eou

Unified:
cargo run --release --example shared_model ./unified audio.wav unified

---

Nemotron (600M): https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b
EOU (120M): https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx
Unified: https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/tree/main
*/

use parakeet_rs::{
    Nemotron, NemotronHandle, ParakeetEOU, ParakeetEOUHandle, ParakeetUnified,
    ParakeetUnifiedHandle,
};

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }
    Ok(audio)
}

fn run_nemotron(model_dir: &str, audio: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let handle = NemotronHandle::load(model_dir, None)?;
    let mut a = Nemotron::from_shared(&handle);
    let mut b = Nemotron::from_shared(&handle);

    let chunk_size = 8960; // 560 ms at 16 kHz
    for chunk_data in audio.chunks(chunk_size) {
        let mut chunk = chunk_data.to_vec();
        chunk.resize(chunk_size, 0.0);
        a.transcribe_chunk(&chunk)?;
        b.transcribe_chunk(&chunk)?;
    }

    println!("A: {}", a.get_transcript());
    println!("B: {}", b.get_transcript());
    assert_eq!(
        a.get_transcript(),
        b.get_transcript(),
        "shared model must be deterministic"
    );
    println!("same");
    Ok(())
}

fn run_eou(model_dir: &str, audio: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let handle = ParakeetEOUHandle::load(model_dir, None)?;
    let mut a = ParakeetEOU::from_shared(&handle);
    let mut b = ParakeetEOU::from_shared(&handle);

    let chunk_size = 2560;
    let mut a_text = String::new();
    let mut b_text = String::new();
    for chunk_data in audio.chunks(chunk_size) {
        let chunk: Vec<f32> = if chunk_data.len() < chunk_size {
            let mut p = chunk_data.to_vec();
            p.resize(chunk_size, 0.0);
            p
        } else {
            chunk_data.to_vec()
        };
        a_text.push_str(&a.transcribe(&chunk, false)?);
        b_text.push_str(&b.transcribe(&chunk, false)?);
    }

    println!("A: {}", a_text.trim());
    println!("B: {}", b_text.trim());
    assert_eq!(a_text, b_text, "shared model must be deterministic");
    println!("same");
    Ok(())
}

fn run_unified(model_dir: &str, audio: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let handle = ParakeetUnifiedHandle::load(model_dir, None)?;
    let mut a = ParakeetUnified::from_shared(&handle);
    let mut b = ParakeetUnified::from_shared(&handle);

    let chunk_size = a.streaming_config().chunk_samples();
    for chunk_data in audio.chunks(chunk_size) {
        a.transcribe_chunk(chunk_data)?;
        b.transcribe_chunk(chunk_data)?;
    }
    a.flush()?;
    b.flush()?;

    println!("A: {}", a.get_transcript());
    println!("B: {}", b.get_transcript());
    assert_eq!(
        a.get_transcript(),
        b.get_transcript(),
        "shared model must be deterministic"
    );
    println!("same");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: shared_model <model_dir> <audio.wav> [eou|unified]");
        std::process::exit(1);
    }
    let model_dir = &args[1];
    let audio_path = &args[2];
    let variant = args.get(3).map(String::as_str).unwrap_or("nemotron");

    let audio = load_wav(audio_path)?;

    match variant {
        "eou" => run_eou(model_dir, &audio),
        "unified" => run_unified(model_dir, &audio),
        _ => run_nemotron(model_dir, &audio),
    }
}
