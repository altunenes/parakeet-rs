/*
Multi-talker streaming ASR with speaker-attributed transcription.

This example combines:
- Sortformer v2: Provides streaming speaker diarisation (4 speakers max)
- Multitalker Parakeet: Speaker kernel injection for per-speaker ASR

Each speaker gets an independent encoder cache and decoder state.
The Sortformer's raw speaker activity probabilities are used as masks
injected into the encoder, enabling per-speaker transcription even
during overlapping speech.

Download models:
- Multitalker ASR: encoder.onnx, decoder_joint.onnx, tokenizer.model
  (exported via conversion_scripts/export_multitalker.py)
- Sortformer v2: diar_streaming_sortformer_4spk-v2.onnx
  https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx

Usage:
  cargo run --release --example multitalker --features multitalker -- \
    <audio.wav> <asr_model_dir> <sortformer.onnx> [options]

Options:
  --max-speakers N     Maximum speakers to track, 1-4 (default: 4)
  --latency MODE       normal, low, very-low, ultra (default: normal)
  --threshold F        Speaker activity threshold, 0.0-1.0 (default: 0.3)
  --batch              Use non-streaming batch transcription

Latency modes: normal (1.12s), low (0.56s), very-low (0.16s), ultra (0.08s)
*/

#[cfg(feature = "multitalker")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{LatencyMode, MultitalkerASR};
    use std::env;
    use std::io::Write;
    use std::time::Instant;

    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!(
            "Usage: {} <audio.wav> <asr_model_dir> <sortformer.onnx> [options]",
            args[0]
        );
        eprintln!();
        eprintln!("  audio.wav        - 16kHz mono WAV file");
        eprintln!("  asr_model_dir    - Directory with encoder.onnx, decoder_joint.onnx, tokenizer.model");
        eprintln!("  sortformer.onnx  - Sortformer v2 ONNX model");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --max-speakers N   Maximum speakers to track, 1-4 (default: 4)");
        eprintln!("  --latency MODE     normal, low, very-low, ultra (default: normal)");
        eprintln!("  --threshold F      Speaker activity threshold, 0.0-1.0 (default: 0.3)");
        eprintln!("  --batch            Use non-streaming batch transcription");
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let asr_model_dir = &args[2];
    let sortformer_path = &args[3];

    // Parse optional flags
    let mut max_speakers: Option<usize> = None;
    let mut latency_mode: Option<LatencyMode> = None;
    let mut activity_threshold: Option<f32> = None;
    let mut batch_mode = false;

    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--max-speakers" => {
                i += 1;
                max_speakers = Some(
                    args.get(i)
                        .ok_or("--max-speakers requires a value")?
                        .parse()
                        .map_err(|_| format!("Invalid max_speakers: {}", args[i]))?,
                );
            }
            "--latency" => {
                i += 1;
                let s = args.get(i).ok_or("--latency requires a value")?;
                latency_mode = Some(match s.as_str() {
                    "normal" => LatencyMode::Normal,
                    "low" => LatencyMode::Low,
                    "very-low" => LatencyMode::VeryLow,
                    "ultra" => LatencyMode::Ultra,
                    _ => {
                        return Err(
                            format!("Unknown latency mode: {s}. Use: normal, low, very-low, ultra")
                                .into(),
                        )
                    }
                });
            }
            "--threshold" => {
                i += 1;
                activity_threshold = Some(
                    args.get(i)
                        .ok_or("--threshold requires a value")?
                        .parse()
                        .map_err(|_| format!("Invalid threshold: {}", args[i]))?,
                );
            }
            "--batch" => {
                batch_mode = true;
            }
            other => return Err(format!("Unknown option: {other}").into()),
        }
        i += 1;
    }

    // Load audio
    println!("Loading audio: {audio_path}");
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

    let duration = audio.len() as f32 / 16000.0;
    println!(
        "Audio: {:.1}s, {} samples, {} Hz",
        duration,
        audio.len(),
        spec.sample_rate
    );

    // Load models
    println!("Loading multitalker ASR model from: {asr_model_dir}");
    println!("Loading Sortformer model from: {sortformer_path}");
    let mut model = MultitalkerASR::from_pretrained(asr_model_dir, sortformer_path, None)?;

    // Apply configuration
    if let Some(n) = max_speakers {
        model.set_max_speakers(n);
    }
    if let Some(mode) = latency_mode {
        model.set_latency_mode(mode);
    }
    if let Some(t) = activity_threshold {
        model.set_activity_threshold(t);
    }

    let config = model.multitalker_config();
    println!(
        "Config: max_speakers={}, latency={:?} ({:.2}s chunks, {} samples/chunk), threshold={:.2}",
        config.max_speakers,
        config.latency_mode,
        config.latency_mode.latency_secs(),
        model.chunk_audio_samples(),
        config.activity_threshold,
    );

    if batch_mode {
        // Non-streaming: process all audio at once
        println!("\nBatch transcription:");
        println!("{}", "=".repeat(60));

        let transcripts = model.transcribe_audio_multitalker(&audio)?;
        print_transcripts(&transcripts);
    } else {
        // Streaming: process chunk by chunk
        let chunk_samples = model.chunk_audio_samples();
        println!("\nStreaming ({chunk_samples} samples per chunk):");
        println!("{}", "=".repeat(60));

        for chunk in audio.chunks(chunk_samples) {
            let chunk_vec = if chunk.len() < chunk_samples {
                let mut p = chunk.to_vec();
                p.resize(chunk_samples, 0.0);
                p
            } else {
                chunk.to_vec()
            };

            let results = model.transcribe_chunk(&chunk_vec)?;
            for r in &results {
                println!("[Speaker {}] {}", r.speaker_id, r.text);
                std::io::stdout().flush()?;
            }
        }

        // Flush with silence
        let flush_chunk = vec![0.0f32; chunk_samples];
        for _ in 0..3 {
            let results = model.transcribe_chunk(&flush_chunk)?;
            for r in &results {
                println!("[Speaker {}] {}", r.speaker_id, r.text);
            }
        }

        println!("\n{}", "=".repeat(60));
        println!("Final transcripts:");
        let transcripts = model.get_transcripts();
        print_transcripts(&transcripts);
    }

    // Tip: for readable multi-speaker output, group words into sentences
    // (split at . ? !) and sort sentences by mean timestamp across speakers.

    let elapsed = start_time.elapsed();
    println!(
        "\nCompleted in {:.2}s (audio: {:.2}s, speed-up: {:.2}x)",
        elapsed.as_secs_f32(),
        duration,
        duration / elapsed.as_secs_f32()
    );

    Ok(())
}

#[cfg(feature = "multitalker")]
fn print_transcripts(transcripts: &[parakeet_rs::SpeakerTranscript]) {
    for transcript in transcripts {
        println!("  Speaker {}: {}", transcript.speaker_id, transcript.text);
        for w in &transcript.words {
            println!(
                "    [{:.2}s - {:.2}s] {}",
                w.start_secs, w.end_secs, w.word
            );
        }
    }
}

#[cfg(not(feature = "multitalker"))]
fn main() {
    eprintln!("This example requires the 'multitalker' feature.");
    eprintln!("Run with: cargo run --example multitalker --features multitalker -- <audio.wav> <model_dir> <sortformer.onnx>");
}
