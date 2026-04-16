/// Benchmark shared NemotronModel between two instances.
///
/// Usage:
///   cargo run --release --example shared_model <model_dir> [audio_a.wav] [audio_b.wav]
///
/// Modes:
///   1 arg:  synthetic test (silence vs tone)
///   2 args: same audio on both instances (determinism check)
///   3 args: different audio per instance (state isolation check)
///
/// Measures: RSS memory, load time, per-chunk latency, total RTF.

use parakeet_rs::Nemotron;
use std::env;
use std::sync::Arc;
use std::time::Instant;

fn rss_mb() -> f64 {
    let pid = std::process::id();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
    }
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok(if spec.channels > 1 {
        audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        audio
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: shared_model <model_dir> [audio_a.wav] [audio_b.wav]");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let audio_path_a = args.get(2).map(|s| s.as_str());
    let audio_path_b = args.get(3).map(|s| s.as_str());

    let rss_baseline = rss_mb();
    println!("=== Shared NemotronModel Benchmark ===\n");
    println!("RSS baseline: {rss_baseline:.1} MB");

    // ── Phase 1: Load model once ────────────────────────────────────────
    let load_start = Instant::now();
    let (shared_model, shared_vocab) = Nemotron::load_model(model_dir, None)?;
    let load_ms = load_start.elapsed().as_millis();
    let rss_after_load = rss_mb();
    println!("Model loaded:  {load_ms}ms | RSS: {rss_after_load:.1} MB (delta: +{:.1} MB)",
        rss_after_load - rss_baseline);

    // ── Phase 2: Create shared instances ────────────────────────────────
    let mut instance_a = Nemotron::from_shared_model(
        Arc::clone(&shared_model),
        Arc::clone(&shared_vocab),
    );
    let mut instance_b = Nemotron::from_shared_model(
        Arc::clone(&shared_model),
        Arc::clone(&shared_vocab),
    );
    let rss_after_instances = rss_mb();
    println!("2 instances:   Arc refcount={} | RSS: {rss_after_instances:.1} MB (delta: +{:.1} MB)",
        Arc::strong_count(&shared_model),
        rss_after_instances - rss_after_load);

    // ── Phase 3: Feed audio ─────────────────────────────────────────────
    let chunk_size = 8960; // 560ms at 16kHz

    if let Some(path_a) = audio_path_a {
        let audio_a = load_wav(path_a)?;
        let path_b = audio_path_b.unwrap_or(path_a);
        let audio_b = load_wav(path_b)?;
        let two_files = audio_path_b.is_some();

        let duration_a = audio_a.len() as f32 / 16000.0;
        let duration_b = audio_b.len() as f32 / 16000.0;
        println!("\nInstance A: {path_a} ({duration_a:.1}s)");
        println!("Instance B: {path_b} ({duration_b:.1}s)");

        // Track per-chunk latency
        let mut chunk_times_a: Vec<f64> = Vec::new();
        let mut chunk_times_b: Vec<f64> = Vec::new();

        let total_start = Instant::now();
        let max_chunks = (audio_a.len() / chunk_size).max(audio_b.len() / chunk_size) + 1;
        for i in 0..max_chunks {
            let offset = i * chunk_size;

            if offset < audio_a.len() {
                let end = (offset + chunk_size).min(audio_a.len());
                let mut chunk = audio_a[offset..end].to_vec();
                chunk.resize(chunk_size, 0.0);
                let t = Instant::now();
                instance_a.transcribe_chunk(&chunk)?;
                chunk_times_a.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            if offset < audio_b.len() {
                let end = (offset + chunk_size).min(audio_b.len());
                let mut chunk = audio_b[offset..end].to_vec();
                chunk.resize(chunk_size, 0.0);
                let t = Instant::now();
                instance_b.transcribe_chunk(&chunk)?;
                chunk_times_b.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let total_ms = total_start.elapsed().as_millis();
        let rss_after_infer = rss_mb();

        let transcript_a = instance_a.get_transcript();
        let transcript_b = instance_b.get_transcript();

        // ── Results ─────────────────────────────────────────────────────
        println!("\n--- Instance A transcript ---");
        println!("{transcript_a}");
        println!("\n--- Instance B transcript ---");
        println!("{transcript_b}");

        // Latency stats
        let avg_a = chunk_times_a.iter().sum::<f64>() / chunk_times_a.len().max(1) as f64;
        let max_a = chunk_times_a.iter().cloned().fold(0.0f64, f64::max);
        let avg_b = chunk_times_b.iter().sum::<f64>() / chunk_times_b.len().max(1) as f64;
        let max_b = chunk_times_b.iter().cloned().fold(0.0f64, f64::max);
        let max_audio_s = duration_a.max(duration_b);

        println!("\n=== Benchmark Results ===");
        println!("Total inference: {total_ms}ms for {max_audio_s:.1}s audio");
        println!("RTF (combined):  {:.2}x real-time", max_audio_s * 1000.0 / total_ms as f32);
        println!("Chunk latency A: avg={avg_a:.1}ms  max={max_a:.1}ms  (budget=560ms)");
        println!("Chunk latency B: avg={avg_b:.1}ms  max={max_b:.1}ms  (budget=560ms)");
        println!("RSS after infer: {rss_after_infer:.1} MB");
        println!("Memory (model):  {:.1} MB (single load)", rss_after_load - rss_baseline);
        println!("Memory (states): {:.1} MB (2 instances overhead)", rss_after_instances - rss_after_load);

        // ── Correctness gate ────────────────────────────────────────────
        if two_files {
            if transcript_a != transcript_b {
                println!("\n✓ PASS: Different audio → different transcripts (state isolation confirmed)");
            } else {
                println!("\n⚠ WARNING: Different audio produced identical transcripts");
            }
        } else if transcript_a == transcript_b {
            println!("\n✓ PASS: Same audio → identical transcripts (determinism confirmed)");
        } else {
            println!("\n✗ FAIL: Same audio → different transcripts (state contamination!)");
            std::process::exit(1);
        }
    } else {
        // Synthetic test
        println!("\nSynthetic: silence (A) vs tone (B)");
        let silence = vec![0.0f32; chunk_size];
        let tone: Vec<f32> = (0..chunk_size)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        for _ in 0..5 {
            instance_a.transcribe_chunk(&silence)?;
            instance_b.transcribe_chunk(&tone)?;
        }

        println!("A (silence): '{}'", instance_a.get_transcript());
        println!("B (tone):    '{}'", instance_b.get_transcript());
        println!("\n✓ PASS: Independent transcripts");
    }

    // ── Cleanup ─────────────────────────────────────────────────────────
    drop(instance_a);
    drop(instance_b);
    println!("Arc refcount after drop: {}", Arc::strong_count(&shared_model));

    Ok(())
}
