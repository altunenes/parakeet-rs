/// CoreML EP diagnostic: compares CPU vs CoreML configurations.
///
/// Usage:
///   cargo run --release --example coreml_diag --features "sortformer,coreml" <audio.wav>
#[cfg(all(feature = "sortformer", feature = "coreml"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ort::ep::coreml::{ComputeUnits, CoreML};
    use ort::ep::CPU;
    use ort::session::builder::SessionBuilder;
    use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
    use parakeet_rs::{ExecutionConfig, ExecutionProvider};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let audio_path = args
        .get(1)
        .expect("Usage: coreml_diag <audio.wav> [model.onnx]");
    let model_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("diar_streaming_sortformer_4spk-v2.onnx");

    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let duration = audio.len() as f32 / spec.sample_rate as f32 / spec.channels as f32;
    println!(
        "Audio: {:.1}s, {} Hz, {} ch\n",
        duration, spec.sample_rate, spec.channels
    );

    println!(
        "| {:<45} | {:>6} | {:>6} | {:>4} | {:>6} |",
        "Config", "Load", "Infer", "Segs", "RT"
    );
    println!("|{:-<47}|{:-<8}|{:-<8}|{:-<6}|{:-<8}|", "", "", "", "", "");

    let run_config = |label: &str,
                      config: ExecutionConfig|
     -> Result<(), Box<dyn std::error::Error>> {
        let load_start = Instant::now();
        let sf = Sortformer::with_config(model_path, Some(config), DiarizationConfig::callhome());
        match sf {
            Ok(mut sf) => {
                let load_time = load_start.elapsed();
                let infer_start = Instant::now();
                let segs = sf.diarize(audio.clone(), spec.sample_rate, spec.channels)?;
                let infer_time = infer_start.elapsed();
                println!(
                    "| {:<45} | {:>5.2}s | {:>5.2}s | {:>4} | {:>5.1}x |",
                    label,
                    load_time.as_secs_f32(),
                    infer_time.as_secs_f32(),
                    segs.len(),
                    duration / infer_time.as_secs_f32()
                );
            }
            Err(e) => {
                let err_short = format!("{}", e);
                let err_short = if err_short.len() > 40 {
                    format!("{}...", &err_short[..40])
                } else {
                    err_short
                };
                println!("| {:<45} | FAILED: {} |", label, err_short);
            }
        }
        Ok(())
    };

    // 1. CPU baseline
    run_config(
        "CPU only",
        ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cpu),
    )?;

    // 2. ExecutionProvider::CoreML (this PR -- reverted to CPUAndGPU)
    run_config(
        "ExecutionProvider::CoreML (this PR)",
        ExecutionConfig::new().with_execution_provider(ExecutionProvider::CoreML),
    )?;

    // 3. CoreML: CPUAndGPU (same as PR, explicit)
    run_config(
        "CoreML: CPUAndGPU (NeuralNetwork)",
        ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cpu)
            .with_custom_configure(|builder: SessionBuilder| {
                builder.with_execution_providers([
                    CoreML::default()
                        .with_compute_units(ComputeUnits::CPUAndGPU)
                        .build(),
                    CPU::default().build().error_on_failure(),
                ])
            }),
    )?;

    // 4. CoreML: All (enables ANE)
    run_config(
        "CoreML: All (ANE enabled)",
        ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cpu)
            .with_custom_configure(|builder: SessionBuilder| {
                builder.with_execution_providers([
                    CoreML::default()
                        .with_compute_units(ComputeUnits::All)
                        .build(),
                    CPU::default().build().error_on_failure(),
                ])
            }),
    )?;

    // 5. CoreML: CPUAndNeuralEngine
    run_config(
        "CoreML: CPUAndNeuralEngine",
        ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cpu)
            .with_custom_configure(|builder: SessionBuilder| {
                builder.with_execution_providers([
                    CoreML::default()
                        .with_compute_units(ComputeUnits::CPUAndNeuralEngine)
                        .build(),
                    CPU::default().build().error_on_failure(),
                ])
            }),
    )?;

    // 6. CoreML: CPUOnly
    run_config(
        "CoreML: CPUOnly",
        ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cpu)
            .with_custom_configure(|builder: SessionBuilder| {
                builder.with_execution_providers([
                    CoreML::default()
                        .with_compute_units(ComputeUnits::CPUOnly)
                        .build(),
                    CPU::default().build().error_on_failure(),
                ])
            }),
    )?;

    println!("\nNote: CoreML claims nodes but runs them on CPU due to dynamic input shapes");
    println!("in the ONNX model. This adds overhead vs the ort CPU EP baseline.");
    println!("Re-exporting the model with fixed shapes would be needed for ANE/GPU dispatch.");

    Ok(())
}

#[cfg(not(all(feature = "sortformer", feature = "coreml")))]
fn main() {
    eprintln!(
        "Requires: cargo run --release --example coreml_diag --features \"sortformer,coreml\" <audio.wav>"
    );
}
