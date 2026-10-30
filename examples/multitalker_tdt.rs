/*
Multitalker TDT: one transcript per speaker, even when they talk over each other.

  cargo run --release --example multitalker_tdt --features sortformer -- \
    <audio.wav> <multitalker_tdt_dir> [nemotron3_diar_v3.onnx]

Audio: 16 kHz WAV (stereo is mixed down).
*/

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("This example requires the 'sortformer' feature: --features sortformer");
}

#[cfg(feature = "sortformer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{ExecutionConfig, MultitalkerTDT};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: multitalker_tdt <audio.wav> <multitalker_tdt_dir> [diar.onnx]";
    let audio_path = args.get(1).ok_or(usage)?;
    let model_dir = args.get(2).ok_or(usage)?;
    let diar_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("nemotron3_diar_v3.onnx");

    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };

    let env = parakeet_rs::ort::init().build()?;
    let config = ExecutionConfig::new().with_environment(env);
    let mut model = MultitalkerTDT::from_pretrained(model_dir, diar_path, Some(config))?;
    let started = Instant::now();
    let utterances = model.transcribe(&audio, spec.sample_rate, spec.channels)?;
    let seconds = audio.len() as f32 / spec.channels as f32 / spec.sample_rate as f32;
    println!(
        "{audio_path}: {seconds:.1}s, transcribed in {:.1}s\n",
        started.elapsed().as_secs_f32()
    );
    for u in &utterances {
        println!(
            "[{:7.2} - {:7.2}] speaker_{}: {}",
            u.start, u.end, u.speaker_id, u.text
        );
    }
    Ok(())
}
