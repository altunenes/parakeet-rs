/*
Multi-talker streaming ASR with speaker-attributed transcription.

Combines Sortformer diarisation with speaker-kernel-injected ASR encoding
to produce per-speaker transcriptions with word-level timestamps.

Download models:
- Multitalker ASR: encoder.onnx, decoder_joint.onnx, tokenizer.model
  (exported via conversion_scripts/export_multitalker.py)
- Sortformer v2: diar_streaming_sortformer_4spk-v2.onnx
  https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx

Usage:
  cargo run --release --example multitalker --features multitalker -- \
    <audio.wav> <asr_model_dir> <sortformer.onnx> [max_speakers] [latency]

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
            "Usage: {} <audio.wav> <asr_model_dir> <sortformer.onnx> [max_speakers] [latency]",
            args[0]
        );
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let asr_model_dir = &args[2];
    let sortformer_path = &args[3];

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

    let duration = audio.len() as f32 / 16000.0;

    // Load models
    let mut model = MultitalkerASR::from_pretrained(asr_model_dir, sortformer_path, None)?;

    if let Some(max_spk_str) = args.get(4) {
        let max_spk: usize = max_spk_str
            .parse()
            .map_err(|_| format!("Invalid max_speakers: {max_spk_str}"))?;
        model.set_max_speakers(max_spk);
    }

    if let Some(latency_str) = args.get(5) {
        let mode = match latency_str.as_str() {
            "normal" => LatencyMode::Normal,
            "low" => LatencyMode::Low,
            "very-low" => LatencyMode::VeryLow,
            "ultra" => LatencyMode::Ultra,
            _ => return Err(format!("Unknown latency: {latency_str}").into()),
        };
        model.set_latency_mode(mode);
    }

    // Stream audio
    let chunk_samples = model.chunk_audio_samples();
    print!("Streaming: ");

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
            print!("[Speaker {}] {}", r.speaker_id, r.text);
            std::io::stdout().flush()?;
        }
    }

    // Flush with silence
    let flush_chunk = vec![0.0f32; chunk_samples];
    for _ in 0..3 {
        let results = model.transcribe_chunk(&flush_chunk)?;
        for r in &results {
            print!("[Speaker {}] {}", r.speaker_id, r.text);
        }
    }

    // Final transcripts with word-level timestamps
    println!("\n\nFinal transcripts:");
    for transcript in model.get_transcripts() {
        println!("  Speaker {}: {}", transcript.speaker_id, transcript.text);
        for w in &transcript.words {
            println!("    [{:.2}s - {:.2}s] {}", w.start_secs, w.end_secs, w.word);
        }
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

#[cfg(not(feature = "multitalker"))]
fn main() {
    eprintln!("This example requires the 'multitalker' feature.");
    eprintln!("Run with: cargo run --example multitalker --features multitalker -- <audio.wav> <model_dir> <sortformer.onnx>");
}
