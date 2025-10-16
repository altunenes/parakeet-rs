# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Rust bindings for NVIDIA's Parakeet ASR model via ONNX Runtime.


GPU support (optional):
```toml
parakeet-rs = { version = "0.1", features = ["cuda"] }
```

Note: CoreML often doesn't work with this model - stick w/ CPU or CUDA. But its incredible fast in my Mac M3 16gb compared to Whisper metal :-)

### Usage

```rust
use parakeet_rs::Parakeet;

let mut parakeet = Parakeet::from_pretrained(".")?;
let result = parakeet.transcribe("audio.wav")?;
println!("{}", result.text);

// Access token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

GPU:
```rust
use parakeet_rs::{Parakeet, ExecutionProvider, ExecutionConfig};

let config = ExecutionConfig::new()
    .with_execution_provider(ExecutionProvider::Cuda);

let mut parakeet = Parakeet::from_pretrained_with_config(".", config)?;
```

### Model Files

Put these in your working directory:
- `model.onnx` / `model.onnx_data`
- `config.json`
- `preprocessor_config.json`
- `tokenizer.json` / `tokenizer_config.json`
- `special_tokens_map.json`

Get the model from HuggingFace [here](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx). 

### Audio Format

- WAV files, 16kHz, mono
- 16-bit PCM or 32-bit float

### Examples

Basic:
```bash
cargo run --example transcribe audio.wav
```

w/ speaker diarization (needs pyannote models):
```bash
cargo run --example pyannote audio.wav
```

### API

```rust
// Load model
let mut parakeet = Parakeet::from_pretrained(".")?;

// Transcribe (returns text + token timestamps)
let result = parakeet.transcribe("audio.wav")?;
println!("{}", result.text);

// Access timestamps
for token in &result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}

// Batch
let results = parakeet.transcribe_batch(&["audio1.wav", "audio2.wav"])?;
for result in results {
    println!("{}", result.text);
}
```

### What it does

- Transcribes speech to text w/ punctuation & capitalization

**Note**: This uses the CTC-based Parakeet model (`nvidia/parakeet-ctc-0.6b`):
- English only
- Token-level timestamps supported (CTC frame-level output)
- For word-level timestamps & speaker diarization, use with pyannote (see example)

#### License

This Rust codebase: **MIT OR Apache-2.0**

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) are licensed under **CC-BY-4.0** by NVIDIA. This library does not distribute the models. 
