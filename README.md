# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast speech recognition with NVIDIA's Parakeet models via ONNX Runtime.
Note: CoreML doesn't stable with this model - stick w/ CPU (or other GPU EP like CUDA). But its incredible fast in my Mac M3 16gb' CPU compared to Whisper metal! :-)

## Models

**CTC (English-only)**: Fast & accurate
```rust
use parakeet_rs::Parakeet;

let mut parakeet = Parakeet::from_pretrained(".", None)?;
let result = parakeet.transcribe_file("audio.wav")?;
println!("{}", result.text);

// Or transcribe in-memory audio
// let result = parakeet.transcribe_samples(audio, 16000, 1)?;

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**TDT (Multilingual)**: 25 languages with auto-detection
```rust
use parakeet_rs::ParakeetTDT;

let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;
let result = parakeet.transcribe_file("audio.wav")?;
println!("{}", result.text);

// Or transcribe in-memory audio (sample_rate: Hz, channels: 1=mono, 2=stereo)
// let result = parakeet.transcribe_samples(audio, 16000, 1)?;

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

## Setup

**CTC**: Download from [HuggingFace](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx): `model.onnx`, `model.onnx_data`, `tokenizer.json`

**TDT**: Download from [HuggingFace](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx): `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt`

Quantized versions available (int8). All files must be in the same directory.

GPU support (auto-falls back to CPU if fails):
```toml
parakeet-rs = { version = "0.1", features = ["cuda"] }  # or tensorrt, webgpu, directml, rocm
```

```rust
use parakeet_rs::{Parakeet, ExecutionConfig, ExecutionProvider};

let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
let mut parakeet = Parakeet::from_pretrained(".", Some(config))?;
```

## Features

- [CTC: English with punctuation & capitalization](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [TDT: Multilingual (auto lang detection) ](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Token-level timestamps
- Speaker diarization: see `examples/pyannote.rs`

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)

## License

Code: MIT OR Apache-2.0

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) are licensed under **CC-BY-4.0** by NVIDIA. This library does not distribute the models.
