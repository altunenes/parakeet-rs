# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast speech recognition with NVIDIA's Parakeet models via ONNX Runtime.
Note: CoreML doesn't stable with this model - stick w/ CPU (or other GPU EP). But its incredible fast in my Mac M3 16gb' CPU compared to Whisper metal! :-)

## Models

**CTC (English-only)**:
```rust
use parakeet_rs::{Parakeet, Transcriber, TimestampMode};

let mut parakeet = Parakeet::from_pretrained(".", None)?;

// Load and transcribe audio (see examples/raw.rs for full example)
let result = parakeet.transcribe_samples(audio, 1600, 1, Some(TimestampMode::Words))?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**TDT (Multilingual)**: 25 languages with auto-detection
```rust
use parakeet_rs::{ParakeetTDT, Transcriber, TimestampMode};

let mut parakeet = ParakeetTDT::from_pretrained("./tdt", None)?;
let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Sentences))?;
println!("{}", result.text);

// Token-level timestamps
for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```

**EOU (Streaming)**: Real-time ASR with end-of-utterance detection
```rust
use parakeet_rs::ParakeetEOU;

let mut parakeet = ParakeetEOU::from_pretrained("./eou", None)?;

// Prepare your audio (Vec<f32>, 16kHz mono, normalized)
let audio: Vec<f32> = /* your audio samples */;

// Process in 160ms chunks for streaming
const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = parakeet.transcribe(chunk, false)?;
    print!("{}", text);
}
```

**Nemotron (Streaming)**: Cache-aware streaming ASR with punctuation
```rust
use parakeet_rs::Nemotron;

let mut model = Nemotron::from_pretrained("./nemotron", None)?;

// Process in 560ms chunks for streaming
const CHUNK_SIZE: usize = 8960; // 560ms at 16kHz
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = model.transcribe_chunk(chunk)?;
    print!("{}", text);
}
```

**Sortformer v2 & v2.1 (Speaker Diarization)**: Streaming 4-speaker diarization
```toml
parakeet-rs = { version = "0.2", features = ["sortformer"] }
```
```rust
use parakeet_rs::sortformer::{Sortformer, DiarizationConfig};

let mut sortformer = Sortformer::with_config(
    "diar_streaming_sortformer_4spk-v2.onnx", // or v2.1.onnx
    None,
    DiarizationConfig::callhome(),  // or dihard3(),custom()
)?;
let segments = sortformer.diarize(audio, 16000, 1)?;
for seg in segments {
    println!("Speaker {} [{:.2}s - {:.2}s]", seg.speaker_id, seg.start, seg.end);
}
```
See `examples/diarization.rs` for combining with TDT transcription.

## Setup

**CTC**: Download from [HuggingFace](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main/onnx): `model.onnx`, `model.onnx_data`, `tokenizer.json`

**TDT**: Download from [HuggingFace](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx): `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt`

**EOU**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx): `encoder.onnx`, `decoder_joint.onnx`, `tokenizer.json`

**Nemotron**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b): `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model` (*[int8](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8) / [int4](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int4)*)

**Diarization (Sortformer v2 & v2.1)**: Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/tree/main): `diar_streaming_sortformer_4spk-v2.onnx` or `v2.1.onnx`.

Quantized versions available (int8). All files must be in the same directory.

GPU support (auto-falls back to CPU if fails):
```toml
parakeet-rs = { version = "0.3", features = ["cuda"] }  # or tensorrt, webgpu, directml, migraphx or other ort supported EPs (check cargo features)
```

```rust
use parakeet_rs::{Parakeet, ExecutionConfig, ExecutionProvider};

let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
let mut parakeet = Parakeet::from_pretrained(".", Some(config))?;
```

Advanced session configuration via [ort SessionBuilder](https://docs.rs/ort/latest/ort/session/builder/struct.SessionBuilder.html):
```rust
let config = ExecutionConfig::new()
    .with_custom_configure(|builder| builder.with_memory_pattern(false));
```

## Features

- [CTC: English with punctuation & capitalization](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
- [TDT: Multilingual (auto lang detection)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [EOU: Streaming ASR with end-of-utterance detection](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- [Nemotron: Cache aware streaming ASR (600M params,EN only)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Sortformer v2 & v2.1: Streaming speaker diarization (up to 4 speakers)](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) NOTE: you can also download v2.1 model same way.
- Token-level timestamps (CTC, TDT)

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)
- CTC/TDT models have ~4-5 minute audio length limit. For longer files, use streaming models or split into chunks

## License

Code: MIT OR Apache-2.0

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) by NVIDIA. This library does not distribute the models.
