//! Run after downloading the verified model with scripts/download_orukeet.py.
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExecutionConfig::new().with_environment(parakeet_rs::ort::init().build()?);
    let mut args = std::env::args_os().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: orukeet MODEL_DIR AUDIO.wav [AUDIO.wav ...]")?;
    let files: Vec<_> = args.collect();
    if files.is_empty() {
        return Err("provide at least one WAV file".into());
    }
    let mut model = ParakeetTDT::from_pretrained(model_dir, Some(config))?;
    for file in files {
        let result = model.transcribe_file(&file, Some(TimestampMode::Words))?;
        println!(
            "{}",
            serde_json::json!({"file": file.to_string_lossy(), "text": result.text})
        );
    }
    Ok(())
}
