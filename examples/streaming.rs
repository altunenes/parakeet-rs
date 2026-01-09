/*
Streaming ASR transcription (real-time, cache-aware stateful)

Nemotron (default):
cargo run --release --example streaming 6_speakers.wav

EOU:
cargo run --release --example streaming 6_speakers.wav eou

---

Nemotron (600M, 24 layers):
- Download: https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b
- Files: encoder.onnx, encoder.onnx.data, decoder_joint.onnx, tokenizer.model
- 560ms chunks

EOU (120M, 17 layers):
- Download: https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx
- Files: encoder.onnx, decoder_joint.onnx, tokenizer.json
- 160ms chunks, no punctuation/capitalization

Additional notes:
let reset_on_eou: bool = false;
I must admit that this is not work very well on my real world tests :/
*/

use parakeet_rs::{Nemotron, ParakeetEOU};
use std::env;
use std::io::Write;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();

    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "6_speakers.wav"
    };

    let use_eou = args.len() > 2 && args[2] == "eou";

    // Load audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
    }

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }

    // Normalize
    let max_val = audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_val > 1e-6 {
        for s in &mut audio {
            *s /= max_val + 1e-5;
        }
    }

    let duration = audio.len() as f32 / 16000.0;

    if use_eou {
        // EOU model
        let mut model = ParakeetEOU::from_pretrained("./fullstr", None)?;
        let chunk_size = 2560; // 160ms

        print!("Streaming: ");
        let mut full_text = String::new();

        for chunk in audio.chunks(chunk_size) {
            let text = model.transcribe(&chunk.to_vec(), false)?;
            if !text.is_empty() {
                print!("{}", text);
                std::io::stdout().flush()?;
                full_text.push_str(&text);
            }
        }

        // Flush
        for _ in 0..3 {
            let text = model.transcribe(&vec![0.0; chunk_size], false)?;
            if !text.is_empty() {
                print!("{}", text);
                full_text.push_str(&text);
            }
        }

        println!("\n\nFinal: {}", full_text.trim());

        let elapsed = start_time.elapsed();
        println!(
            "Completed in {:.2}s (audio: {:.2}s, RTF: {:.2}x)",
            elapsed.as_secs_f32(),
            duration,
            duration / elapsed.as_secs_f32()
        );
        return Ok(());
    }

    // Nemotron (default)
    let mut model = Nemotron::from_pretrained("./nemotron", None)?;
    let chunk_size = 8960; // 560ms

    print!("Streaming: ");

    for chunk in audio.chunks(chunk_size) {
        let chunk_vec = if chunk.len() < chunk_size {
            let mut p = chunk.to_vec();
            p.resize(chunk_size, 0.0);
            p
        } else {
            chunk.to_vec()
        };

        let text = model.transcribe_chunk(&chunk_vec)?;
        if !text.is_empty() {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }

    // Flush
    for _ in 0..3 {
        let text = model.transcribe_chunk(&vec![0.0; chunk_size])?;
        if !text.is_empty() {
            print!("{}", text);
        }
    }

    println!("\n\nFinal: {}", model.get_transcript());

    let elapsed = start_time.elapsed();
    println!(
        "Completed in {:.2}s (audio: {:.2}s, RTF: {:.2}x)",
        elapsed.as_secs_f32(),
        duration,
        duration / elapsed.as_secs_f32()
    );

    Ok(())
}
