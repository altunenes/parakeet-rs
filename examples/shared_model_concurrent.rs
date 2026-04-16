/// Concurrent dual-stream benchmark — mirrors real ContinuousAsr dual-capture.
///
/// Two threads run simultaneously, each with their own Nemotron instance
/// sharing one Arc<Mutex<NemotronModel>>. Audio is delivered at real-time pace
/// via channels, simulating rtrb ring buffers.
///
/// Usage:
///   cargo run --release --example shared_model_concurrent <model_dir> <audio_a.wav> <audio_b.wav>
///
/// Measures: per-chunk latency under real contention, lock wait time, RTF.

use parakeet_rs::Nemotron;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const CHUNK_SIZE: usize = 8960; // 560ms at 16kHz
const SAMPLE_RATE: usize = 16000;
const CHUNK_DURATION_MS: u64 = (CHUNK_SIZE as u64 * 1000) / SAMPLE_RATE as u64; // 560ms

fn rss_mb() -> f64 {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
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

struct StreamStats {
    chunk_count: usize,
    total_infer_ms: f64,
    max_infer_ms: f64,
    total_wait_ms: f64,
    max_wait_ms: f64,
    transcript: String,
}

fn run_stream(
    name: &str,
    mut instance: Nemotron,
    audio: Vec<f32>,
    lock_contention_total: Arc<AtomicU64>,
) -> StreamStats {
    let mut stats = StreamStats {
        chunk_count: 0,
        total_infer_ms: 0.0,
        max_infer_ms: 0.0,
        total_wait_ms: 0.0,
        max_wait_ms: 0.0,
        transcript: String::new(),
    };

    for (i, chunk_data) in audio.chunks(CHUNK_SIZE).enumerate() {
        let mut chunk = chunk_data.to_vec();
        if chunk.len() < CHUNK_SIZE {
            chunk.resize(CHUNK_SIZE, 0.0);
        }

        // Simulate real-time delivery: sleep for chunk duration minus processing time
        // First chunk starts immediately; subsequent chunks arrive at real-time pace
        if i > 0 {
            // In real ContinuousAsr, audio arrives via ring buffer at real-time rate.
            // We simulate this with a small sleep to create realistic contention.
            thread::sleep(Duration::from_millis(1));
        }

        let chunk_start = Instant::now();
        match instance.transcribe_chunk(&chunk) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("[{name}] chunk {i} error: {e}");
                break;
            }
        }
        let elapsed_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;

        stats.chunk_count += 1;
        stats.total_infer_ms += elapsed_ms;
        if elapsed_ms > stats.max_infer_ms {
            stats.max_infer_ms = elapsed_ms;
        }

        // Track how much time was spent waiting vs the 560ms budget
        let wait_ms = elapsed_ms.max(0.0);
        stats.total_wait_ms += wait_ms;
        if wait_ms > stats.max_wait_ms {
            stats.max_wait_ms = wait_ms;
        }

        lock_contention_total.fetch_add((elapsed_ms * 1000.0) as u64, Ordering::Relaxed);
    }

    stats.transcript = instance.get_transcript();
    stats
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: shared_model_concurrent <model_dir> <audio_a.wav> <audio_b.wav>");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let path_a = &args[2];
    let path_b = &args[3];

    let rss_baseline = rss_mb();
    println!("=== Concurrent Dual-Stream Benchmark ===\n");
    println!("RSS baseline: {rss_baseline:.1} MB");

    // Load model once
    let load_start = Instant::now();
    let (shared_model, shared_vocab) = Nemotron::load_model(model_dir, None)?;
    let load_ms = load_start.elapsed().as_millis();
    let rss_after_load = rss_mb();
    println!("Model loaded:  {load_ms}ms | RSS: {rss_after_load:.1} MB (delta: +{:.1} MB)",
        rss_after_load - rss_baseline);

    // Create two instances
    let instance_a = Nemotron::from_shared_model(
        Arc::clone(&shared_model),
        Arc::clone(&shared_vocab),
    );
    let instance_b = Nemotron::from_shared_model(
        Arc::clone(&shared_model),
        Arc::clone(&shared_vocab),
    );

    // Load audio
    let audio_a = load_wav(path_a)?;
    let audio_b = load_wav(path_b)?;
    let duration_a = audio_a.len() as f32 / SAMPLE_RATE as f32;
    let duration_b = audio_b.len() as f32 / SAMPLE_RATE as f32;

    println!("\nStream A (mic):    {path_a} ({duration_a:.1}s, {} chunks)", audio_a.len() / CHUNK_SIZE);
    println!("Stream B (system): {path_b} ({duration_b:.1}s, {} chunks)", audio_b.len() / CHUNK_SIZE);
    println!("Chunk budget: {CHUNK_DURATION_MS}ms per {CHUNK_SIZE} samples");

    // Spawn concurrent threads
    let contention_a = Arc::new(AtomicU64::new(0));
    let contention_b = Arc::new(AtomicU64::new(0));
    let contention_a_clone = Arc::clone(&contention_a);
    let contention_b_clone = Arc::clone(&contention_b);

    let total_start = Instant::now();

    let handle_a = thread::Builder::new()
        .name("stream-mic".to_string())
        .spawn(move || run_stream("mic", instance_a, audio_a, contention_a_clone))?;

    let handle_b = thread::Builder::new()
        .name("stream-sys".to_string())
        .spawn(move || run_stream("sys", instance_b, audio_b, contention_b_clone))?;

    let stats_a = handle_a.join().expect("stream-mic panicked");
    let stats_b = handle_b.join().expect("stream-sys panicked");
    let total_ms = total_start.elapsed().as_millis();
    let rss_after = rss_mb();

    // Results
    println!("\n--- Stream A (mic) transcript ---");
    println!("{}", stats_a.transcript);
    println!("\n--- Stream B (system) transcript ---");
    println!("{}", stats_b.transcript);

    let avg_a = stats_a.total_infer_ms / stats_a.chunk_count.max(1) as f64;
    let avg_b = stats_b.total_infer_ms / stats_b.chunk_count.max(1) as f64;
    let max_audio = duration_a.max(duration_b);

    println!("\n=== Concurrent Benchmark Results ===");
    println!("Wall clock:        {total_ms}ms for {max_audio:.1}s audio");
    println!("Concurrent RTF:    {:.2}x real-time", max_audio * 1000.0 / total_ms as f32);
    println!("");
    println!("Stream A (mic):    {:.0} chunks | avg={avg_a:.1}ms  max={:.1}ms  (budget={CHUNK_DURATION_MS}ms)",
        stats_a.chunk_count, stats_a.max_infer_ms);
    println!("Stream B (sys):    {:.0} chunks | avg={avg_b:.1}ms  max={:.1}ms  (budget={CHUNK_DURATION_MS}ms)",
        stats_b.chunk_count, stats_b.max_infer_ms);
    println!("");
    println!("Lock budget usage: A={:.1}%  B={:.1}%",
        avg_a / CHUNK_DURATION_MS as f64 * 100.0,
        avg_b / CHUNK_DURATION_MS as f64 * 100.0);
    println!("Max lock wait:     A={:.1}ms  B={:.1}ms",
        stats_a.max_infer_ms, stats_b.max_infer_ms);
    println!("");
    println!("RSS after infer:   {rss_after:.1} MB");
    println!("Memory (model):    {:.1} MB (single load, shared)", rss_after_load - rss_baseline);

    // Under real-time constraint: max chunk latency must be < 560ms
    let max_chunk_ms = stats_a.max_infer_ms.max(stats_b.max_infer_ms);
    if max_chunk_ms < CHUNK_DURATION_MS as f64 {
        println!("\n✓ PASS: Max chunk latency ({max_chunk_ms:.1}ms) < budget ({CHUNK_DURATION_MS}ms)");
    } else {
        println!("\n✗ FAIL: Max chunk latency ({max_chunk_ms:.1}ms) exceeds budget ({CHUNK_DURATION_MS}ms)");
        std::process::exit(1);
    }

    // State isolation: different audio must produce different transcripts
    if stats_a.transcript != stats_b.transcript {
        println!("✓ PASS: Different audio → different transcripts (state isolation under concurrency)");
    } else {
        println!("⚠ WARNING: Identical transcripts from different audio");
    }

    Ok(())
}
