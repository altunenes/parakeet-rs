# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast English speech recognition with NVIDIA's Parakeet model via ONNX Runtime.
Note: CoreML doesn't work with this model - stick w/ CPU (or other GPU EP like CUDA). But its incredible fast in my Mac M3 16gb' CPU compared to Whisper metal! :-)


```rust
use parakeet_rs::Parakeet;

let mut parakeet = Parakeet::from_pretrained(".")?;
let result = parakeet.transcribe("audio.wav")?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

## Setup

Download from [HuggingFace](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx): `model.onnx`, `model.onnx_data`, `tokenizer.json`

Quantized versions also available (fp16, int8, q4). All 3 files must be in the same directory.

GPU support:
```toml
parakeet-rs = { version = "0.x", features = ["cuda"] }
```

```rust
use parakeet_rs::{ExecutionConfig, ExecutionProvider};

let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
let mut parakeet = Parakeet::from_pretrained_with_config(".", config)?;
```

## Features

- English transcription with punctuation & capitalization
- Token-level timestamps from CTC output
- Batch processing: `transcribe_batch(&["a.wav", "b.wav"])` etc
- See `examples/pyannote.rs` for speaker diarization + transcription.

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)

## License

Code: MIT OR Apache-2.0

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) are licensed under **CC-BY-4.0** by NVIDIA. This library does not distribute the models.
