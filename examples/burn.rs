/*
Models on the burn backend: pure Rust, no ONNX Runtime.

They read the same model files as the ONNX Runtime backend:
  tdt   TDT folder (encoder-model.onnx + .data, decoder_joint-model.onnx, vocab.txt):
        NVIDIA v3 or Moondream's Parakeet Ultra
  ctc   CTC folder (model.onnx + model.onnx_data, tokenizer.json)
  unified  Unified folder (fp32 encoder.onnx + .data, decoder_joint.onnx, tokenizer.model)
  eou   realtime EOU folder (encoder.onnx, decoder_joint.onnx, tokenizer.json)
  nemotron  Nemotron streaming folder (encoder.onnx + .data, decoder_joint.onnx, tokenizer.model),
        English or multilingual
  multitalker  Multitalker folder (fp32 encoder.onnx + .data, decoder_joint.onnx, tokenizer.model);
        5th argument: nemotron3_diar_v3.onnx (needs --features multitalker)
  diar  nemotron3_diar_v3.onnx (Nemotron-3 Diarization; needs --features sortformer),
        optional 5th argument: latency preset offline (default) | low | very-low | ultra

GPU (wgpu: Metal on macOS, DX12/Vulkan on Windows, Vulkan on Linux), without ONNX Runtime:
  cargo run --release --example burn --no-default-features --features wgpu -- tdt audio.wav ./tdt gpu
Apple GPUs: use --features metal instead of wgpu (kernels compiled for Metal, much faster).
Vulkan GPUs: --features vulkan. NVIDIA/AMD: --features burn-cuda / burn-rocm with device cuda / rocm.
CPU:
  cargo run --release --example burn --no-default-features --features burn -- tdt audio.wav ./tdt
Compare with ONNX Runtime (default features plus wgpu; device: cpu | gpu | ort):
  cargo run --release --example burn --features wgpu -- tdt audio.wav ./tdt ort

The first GPU run on a machine compiles and tunes kernels for each new range of audio lengths,
which takes seconds. Results are cached on disk, so later runs and later launches are fast:
an app can transcribe a short clip at startup to warm up.
*/

use parakeet_rs::{
    ExecutionConfig, ExecutionProvider, Parakeet, ParakeetTDT, TimestampMode, Transcriber,
};
use std::time::Instant;

type Run = Box<dyn FnMut() -> Result<Vec<String>, Box<dyn std::error::Error>>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: burn <tdt|ctc|unified|eou|nemotron|multitalker|diar> <audio.wav> <model path> [cpu|gpu|ort] [diar preset | diar model]";
    let (Some(kind), Some(audio_path), Some(model_path)) = (args.get(1), args.get(2), args.get(3))
    else {
        return Err(usage.into());
    };
    let provider = match args.get(4).map(String::as_str).unwrap_or("cpu") {
        "cpu" => ExecutionProvider::BurnCpu,
        #[cfg(feature = "wgpu")]
        "gpu" => ExecutionProvider::BurnWgpu,
        #[cfg(feature = "burn-cuda")]
        "cuda" => ExecutionProvider::BurnCuda,
        #[cfg(feature = "burn-rocm")]
        "rocm" => ExecutionProvider::BurnRocm,
        // ONNX Runtime on the CPU, to compare (needs the default `ort` feature)
        #[cfg(feature = "ort")]
        "ort" => ExecutionProvider::Cpu,
        other => {
            return Err(format!(
                "unknown device {other}: cpu | gpu (--features wgpu) | ort (default features)"
            )
            .into());
        }
    };

    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    let (rate, channels) = (spec.sample_rate, spec.channels);
    let config = ExecutionConfig::new().with_execution_provider(provider);
    let lines = |tokens: &[parakeet_rs::TimedToken]| -> Vec<String> {
        tokens
            .iter()
            .map(|t| format!("[{:6.2}s - {:6.2}s] {}", t.start, t.end, t.text))
            .collect()
    };

    let started = Instant::now();
    let mut run: Run = match kind.as_str() {
        "tdt" => {
            let mut asr = ParakeetTDT::from_pretrained(model_path, Some(config))?;
            Box::new(move || {
                let mode = Some(TimestampMode::Sentences);
                let result = asr.transcribe_samples(audio.clone(), rate, channels, mode)?;
                Ok(lines(&result.tokens))
            })
        }
        "ctc" => {
            let mut asr = Parakeet::from_pretrained(model_path, Some(config))?;
            Box::new(move || {
                // CTC predicts no punctuation, so words rather than sentences
                let mode = Some(TimestampMode::Words);
                let result = asr.transcribe_samples(audio.clone(), rate, channels, mode)?;
                Ok(lines(&result.tokens))
            })
        }
        "unified" => {
            let mut asr = parakeet_rs::ParakeetUnified::from_pretrained(model_path, Some(config))?;
            Box::new(move || {
                let mode = Some(TimestampMode::Words);
                let result = asr.transcribe_samples(audio.clone(), rate, channels, mode)?;
                Ok(lines(&result.tokens))
            })
        }
        "eou" => {
            let handle = parakeet_rs::ParakeetEOUHandle::from_pretrained(model_path, Some(config))?;
            let mono: Vec<f32> = audio
                .chunks(channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect();
            Box::new(move || {
                // a fresh stream each run; 160 ms chunks, then silence to flush
                let mut asr = parakeet_rs::ParakeetEOU::from_shared(&handle);
                let mut text = String::new();
                for chunk in mono
                    .chunks(2560)
                    .chain(std::iter::repeat_n(&[0.0f32; 2560][..], 3))
                {
                    text.push_str(&asr.transcribe(chunk, false)?);
                }
                Ok(vec![format!("[transcript] {text}")])
            })
        }
        "nemotron" => {
            let mut asr = parakeet_rs::Nemotron::from_pretrained(model_path, Some(config))?;
            // streaming models take 16 kHz mono
            let mono: Vec<f32> = audio
                .chunks(channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect();
            Box::new(move || {
                asr.reset();
                Ok(vec![format!(
                    "[transcript] {}",
                    asr.transcribe_audio(&mono)?
                )])
            })
        }
        #[cfg(feature = "multitalker")]
        "multitalker" => {
            let diar = args
                .get(5)
                .ok_or("multitalker needs the diarization model as 5th argument")?;
            let mut asr =
                parakeet_rs::MultitalkerASR::from_pretrained(model_path, diar, Some(config))?;
            let mono: Vec<f32> = audio
                .chunks(channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect();
            Box::new(move || {
                Ok(asr
                    .transcribe_audio_multitalker(&mono)?
                    .iter()
                    .map(|t| format!("[speaker_{}] {}", t.speaker_id, t.text))
                    .collect())
            })
        }
        #[cfg(feature = "sortformer")]
        "diar" => {
            use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, StreamingProfile};
            let mut diarizer =
                Sortformer::with_config(model_path, Some(config), DiarizationConfig::default())?;
            diarizer.set_profile(match args.get(5).map(String::as_str).unwrap_or("offline") {
                "offline" => StreamingProfile::offline(),
                "low" => StreamingProfile::low_latency(),
                "very-low" => StreamingProfile::very_low_latency(),
                "ultra" => StreamingProfile::ultra_low_latency(),
                other => return Err(format!("unknown preset {other}").into()),
            })?;
            Box::new(move || {
                let segments = diarizer.diarize(audio.clone(), rate, channels)?;
                Ok(segments
                    .iter()
                    .map(|s| {
                        format!(
                            "[{:7.2}s - {:7.2}s] speaker_{}",
                            s.start as f64 / 16_000.0,
                            s.end as f64 / 16_000.0,
                            s.speaker_id
                        )
                    })
                    .collect())
            })
        }
        other => return Err(format!("unknown model {other}; {usage}").into()),
    };
    println!(
        "Loaded {model_path} on {provider:?} in {:.2}s",
        started.elapsed().as_secs_f32()
    );

    // Twice: on the GPU the first run includes kernel compilation and tuning (cached afterwards).
    for i in 1..=2 {
        let started = Instant::now();
        let output = run()?;
        println!("\nRun {i}: {:.2}s", started.elapsed().as_secs_f32());
        if i == 2 {
            for line in output {
                println!("{line}");
            }
        }
    }
    Ok(())
}
