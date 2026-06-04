/*
Streaming ASR transcription (real-time, cache-aware stateful)

Usage:
  cargo run --release --example streaming <audio.wav>               # English Nemotron (./nemotron)
  cargo run --release --example streaming <audio.wav> <lang>        # Multilingual Nemotron (./nemotron_multi)
  cargo run --release --example streaming <audio.wav> eou           # EOU streaming (./fullstr)

Examples:
  cargo run --release --example streaming 6_speakers.wav            # English specialist
  cargo run --release --example streaming swed.wav  sv-SE           # Swedish hint
  cargo run --release --example streaming DENIZ.wav tr-TR           # Turkish hint
  cargo run --release --example streaming clip.wav  auto            # let the model pick the lang
  cargo run --release --example streaming 6_speakers.wav eou        # EOU model instead of Nemotron

Any 2nd arg that is not literally `eou` is treated as a target language code
(`en-US`, `es-ES`, `ja-JP`, `tr-TR`, `auto`, etc. see prompt_dictionary in
src/nemotron.rs for the full list). If a lang is given, the example will load
the multilingual model from ./nemotron_multi. With no lang, it loads the
English-only ./nemotron.

---

Nemotron English-only (600M, 24 layers, vocab 1024):
- Download: https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b
- Files: encoder.onnx, encoder.onnx.data, decoder_joint.onnx, tokenizer.model
- Expects path: ./nemotron
- 560ms chunks

Nemotron multilingual 3.5 (600M, 40 language-locales, vocab 13087):
- Download: https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3.5-asr-streaming-0.6b-onnx
  (or export yourself from https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
  with scripts/export_nemotron_streaming_multilingual.py).
- 40 language-locales documented in NVIDIA's model card across 3 tiers
  (19 transcription-ready, 13 broad-coverage, 8 adaptation-ready that need
  fine-tuning). The prompt dictionary accepts more codes but those extras
  are experimental and not in the model card.
- Files in the same layout (encoder.onnx + .data, decoder_joint.onnx, tokenizer.model)
- Expects path: ./nemotron_multi
- Variant is auto-detected at load time — same `Nemotron::from_pretrained` call.

EOU (120M, 17 layers):
- Download: https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx
- Files: encoder.onnx, decoder_joint.onnx, tokenizer.json
- 160ms chunks, no punctuation/capitalization

Additional notes:
let reset_on_eou: bool = false;
I must admit that this is not work very well on my real world tests :/
*/

use parakeet_rs::{Nemotron, NemotronMode, ParakeetEOU};
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
    // 3rd arg (if not "eou") is treated as target language for the multilingual Nemotron.
    // Ignored for English-only model.
    let target_lang: Option<&str> = if args.len() > 2 && args[2] != "eou" {
        Some(args[2].as_str())
    } else if args.len() > 3 {
        Some(args[3].as_str())
    } else {
        None
    };

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

    // I use the lang hint as a signal here. If you bothered to pass one
    // you obviously want the multilingual model, so I jump straight to
    // ./nemotron_multi. With no hint, I default to the English specialist
    // (more verbatim on English audio) and only fall back to the multilingual
    // dir if that's the only one sitting on disk.
    let model_dir = if target_lang.is_some() {
        "./nemotron_multi"
    } else if std::path::Path::new("./nemotron").is_dir() {
        "./nemotron"
    } else {
        "./nemotron_multi"
    };
    let mut model = Nemotron::from_pretrained(model_dir, None)?;
    let chunk_size = 8960; // 560ms

    match model.mode() {
        NemotronMode::Multilingual => {
            let lang = target_lang.unwrap_or("auto");
            model.set_target_lang(lang)?;
            println!("[multilingual model, target_lang={lang}]");
        }
        NemotronMode::EnglishOnly => {
            if let Some(lang) = target_lang {
                eprintln!("Warning: target_lang='{lang}' ignored, English-only model loaded");
            }
        }
    }

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
