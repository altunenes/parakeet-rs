# parakeet-rs
[![Rust](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/parakeet-rs/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/parakeet-rs.svg)](https://crates.io/crates/parakeet-rs)

Fast speech recognition and diarization with NVIDIA's Parakeet models, on ONNX Runtime or pure Rust with [burn](#pure-rust-backend-burn).

## Models

| Model | Type | Lang | Download | Example |
|---|---|---|---|---|
| [CTC](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | Offline | eng | [parakeet-ctc-0.6b](https://huggingface.co/altunenes/parakeet-rs/tree/main/parakeet-ctc-0.6b) | `raw` |
| [TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Offline, auto language | 25 | [parakeet-tdt-0.6b-v3](https://huggingface.co/altunenes/parakeet-rs/tree/main/parakeet-tdt-0.6b-v3) | `raw` |
| [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) | Offline, Moondream's tuned TDT | 25 | [parakeet-ultra](https://huggingface.co/altunenes/parakeet-rs/tree/main/parakeet-ultra) | `raw` |
| [Orukeet](https://huggingface.co/oruk/orukeet) | Offline, TDT fine tune, int8 | 25 | [see below](#orukeet) | `orukeet` |
| [EOU](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) | Streaming, end of utterance | eng | [realtime_eou_120m-v1-onnx](https://huggingface.co/altunenes/parakeet-rs/tree/main/realtime_eou_120m-v1-onnx) | `streaming` |
| [Nemotron](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | Streaming, verbatim | eng | [nemotron-speech-streaming-en-0.6b](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b) | `streaming` |
| [Nemotron 3.5](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Streaming | 40 | [nemotron-3.5-asr-streaming-0.6b-onnx](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3.5-asr-streaming-0.6b-onnx) | `streaming` |
| [Unified](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | Offline + streaming | eng | [parakeet-unified-en-0.6b](https://huggingface.co/altunenes/parakeet-rs/tree/main/parakeet-unified-en-0.6b) | `unified` |
| [Unified](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) | Offline + streaming | 25 | [nemotron-labs-unified-asr-0.6b](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-labs-unified-asr-0.6b) | `unified` |
| [Multitalker](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) | Streaming, per speaker | eng | [multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/altunenes/parakeet-rs/tree/main/multitalker-parakeet-streaming-0.6b-v1) | `multitalker` |
| **Multitalker TDT** | Offline, per speaker, overlaps ([how](https://altunenes.github.io/posts/multitalker/)) | 25 | [multitalker-tdt-0.6b-v3](https://huggingface.co/altunenes/parakeet-rs/tree/main/multitalker-tdt-0.6b-v3) | `multitalker_tdt` |
| [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Offline, long form | 14 | [onnx-community](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX) | `cohere` |
| [Nemotron-3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | Who speaks when, up to 8 | any | [nemotron-3-diarization](https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3-diarization) | `diarization` |

Download a folder and pass it to `from_pretrained`:
```bash
hf download altunenes/parakeet-rs --include "parakeet-tdt-0.6b-v3/*" --local-dir .
```
Multitalker and Multitalker TDT also need the diarization model. Some folders have int8 too.
Or export them yourself from NeMo with `scripts/export_*.py` (ONNX opset 23); Cohere comes from Optimum.

## Usage

ONNX Runtime (the default backend) needs one environment per process, shared by every model:
```rust
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

let env = parakeet_rs::ort::init().build()?;
let config = ExecutionConfig::new().with_environment(env);

let mut parakeet = ParakeetTDT::from_pretrained("./tdt", Some(config.clone()))?;
let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Sentences))?;
println!("{}", result.text);

for token in result.tokens {
    println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
}
```
CTC is the same with `Parakeet` instead of `ParakeetTDT`. Every other model:

<details>
<summary><b>EOU</b> (streaming)</summary>

```rust
use parakeet_rs::ParakeetEOU;

let mut parakeet = ParakeetEOU::from_pretrained("./eou", Some(config.clone()))?;

// 160 ms chunks, 16kHz mono
const CHUNK_SIZE: usize = 2560;
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = parakeet.transcribe(chunk, false)?;
    print!("{}", text);
}
```
</details>

<details>
<summary><b>Nemotron</b> (streaming, eng or 40)</summary>

Same API for both; the loader detects which one the folder holds. English is verbatim and keeps
`um`, `uh`. 3.5 has [40 languages](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
casing and punctuation, and drops them.

```rust
use parakeet_rs::{Nemotron, NemotronMode};

let mut model = Nemotron::from_pretrained(path, Some(config.clone()))?;

// 3.5: "auto" by default, a known language is more accurate.
if model.mode() == NemotronMode::Multilingual {
    model.set_target_lang("es-ES")?; // also: "ja-JP", "tr-TR", "auto", ...
}

// 560 ms chunks
const CHUNK_SIZE: usize = 8960;
for chunk in audio.chunks(CHUNK_SIZE) {
    let text = model.transcribe_chunk(chunk)?;
    print!("{}", text);
}
```
</details>

<details>
<summary><b>Unified</b> (offline + streaming, eng or 25)</summary>

One model for offline and buffered streaming. The 25 language one takes the audio's language;
`auto` by default, a known language is a bit more accurate.
```rust
use parakeet_rs::{ParakeetUnified, TimestampMode, Transcriber};

let mut model = ParakeetUnified::from_pretrained("./unified", Some(config.clone()))?;
model.set_target_lang("de")?; // 25 language model only: "fr", "pl", "auto", ...

// Offline
let result = model.transcribe_samples(audio.clone(), 16000, 1, Some(TimestampMode::Words))?;

// Streaming: 560 ms chunks, 5.6 s left context by default
let chunk = model.streaming_config().chunk_samples();
for piece in audio.chunks(chunk) {
    print!("{}", model.transcribe_chunk(piece)?);
}
print!("{}", model.flush()?);
```
</details>

<details>
<summary><b>Multitalker</b> (streaming, per speaker, eng)</summary>

```toml
parakeet-rs = { version = "0.4", features = ["multitalker"] }
```
```rust
use parakeet_rs::MultitalkerASR;

let mut model = MultitalkerASR::from_pretrained(
    "./multitalker",             // encoder, decoder, tokenizer
    "nemotron3_diar_v3.onnx",    // Nemotron-3 (Sortformer v3) for diarization
    Some(config.clone()),
)?;

for chunk in audio.chunks(17920) {  // ~1.12s at 16kHz
    let results = model.transcribe_chunk(chunk)?;
    for r in &results {
        println!("[Speaker {}] {}", r.speaker_id, r.text);
    }
}
```
See `examples/multitalker.rs` for latency modes.
</details>

<details>
<summary><b>Multitalker TDT</b> (offline, per speaker, 25)</summary>

parakeet-tdt-0.6b-v3 fine tuned with speaker kernels: one transcript per speaker, even when they
talk over each other. Any audio length.
```toml
parakeet-rs = { version = "0.4", features = ["sortformer"] }
```
```rust
use parakeet_rs::MultitalkerTDT;

let mut model = MultitalkerTDT::from_pretrained(
    "./multitalker-tdt",         // encoder, decoder_joint, vocab.txt
    "nemotron3_diar_v3.onnx",    // Nemotron-3 (Sortformer v3) for diarization
    Some(config.clone()),
)?;

for u in model.transcribe(&audio, 16000, 1)? {
    println!("[{:.2} - {:.2}] Speaker {}: {}", u.start, u.end, u.speaker_id, u.text);
}
```
cpWER with 2 overlapping speakers, plain TDT + diarization → Multitalker TDT: English
([LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)) 36.7% → 4.4%, six other
languages ([MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) mixtures)
39.0% → 10.7%. Single speakers stay as good as plain TDT. Details in the
[blog post](https://altunenes.github.io/posts/multitalker/).
</details>

<details>
<summary><b>Cohere Transcribe</b> (offline, 14)</summary>

Yes, "parakeets🦜" talk about more than just NVIDIA :-P
```toml
parakeet-rs = { version = "0.4", features = ["cohere"] }
```
```rust
use parakeet_rs::CohereASR;

let mut model = CohereASR::from_pretrained("./cohere", Some(config.clone()))?;

// 16kHz mono, long audio is fine
let text = model.transcribe_audio(&audio, "en", true, false)?; // lang, pnc, itn
println!("{}", text);
```
Files: `encoder_model.onnx` (+ `.onnx_data*`), `decoder_model_merged.onnx` (+ `.onnx_data`),
`tokenizer.json` (FP32, FP16, INT8, INT4 variants available).
</details>

<details>
<summary><b>Nemotron-3 Diarization</b> (Sortformer v3, up to 8 speakers)</summary>

```toml
parakeet-rs = { version = "0.4", features = ["sortformer"] }
```
```rust
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, StreamingProfile};

let mut diarizer = Sortformer::with_config(
    "nemotron3_diar_v3.onnx",
    Some(config.clone()),
    DiarizationConfig::default(),
)?;
let segments = diarizer.diarize(audio, 16000, 1)?;
for seg in segments {
    println!("speaker_{} [{:.2}s - {:.2}s]", seg.speaker_id,
        seg.start as f64 / 16_000.0, seg.end as f64 / 16_000.0);
}

// Streaming: diarize_chunk() / feed() keep state across calls.
let segments = diarizer.diarize_chunk(&audio_chunk_16k_mono)?;

// Latency presets from NVIDIA's model card:
// offline() 30.4s (default), low_latency() 1.04s, very_low_latency() 0.64s, ultra_low_latency() 0.32s
diarizer.set_profile(StreamingProfile::low_latency())?;
```
`examples/diarization.rs` (pass `low`, `very-low` or `ultra` for the presets),
`examples/streaming_diarization.rs` for live `feed`/`flush`, and
`scripts/export_diar_sortformer.py` to export with custom streaming parameters.
</details>

<details id="orukeet">
<summary><b>Orukeet</b> (TDT fine tune, int8)</summary>

The [pinned Hugging Face INT8 export](https://huggingface.co/oruk/orukeet/tree/eac739d754bb171287930e6e63386f5b88f8179e/onnx/combined-v0.1.0-int8) works with `ParakeetTDT`. The downloader verifies a release manifest and all required file hashes, including licenses, then prints the cached model directory:

```bash
python3 -m pip install huggingface-hub
model_dir=$(python3 scripts/download_orukeet.py)
cargo run --release --example orukeet -- "$model_dir" audio.wav
# After installation, no network is needed:
model_dir=$(python3 scripts/download_orukeet.py --offline)
```

It runs on the TDT code, audio stays local. The ~672 MB weights are [CC BY-SA 4.0](https://huggingface.co/oruk/orukeet/blob/eac739d754bb171287930e6e63386f5b88f8179e/onnx/combined-v0.1.0-int8/LICENSE-WEIGHTS) (see the downloaded `LICENSE-WEIGHTS`); `NOTICE.md` keeps the NVIDIA attribution. The model card's accuracy numbers are from NeMo, not from this runtime.
</details>

## GPU and backends

GPU with ONNX Runtime (falls back to CPU if it fails):
```toml
parakeet-rs = { version = "0.4", features = ["cuda"] }  # or tensorrt, webgpu, directml, migraphx or other ort supported EPs (check cargo features)
```
```rust
use parakeet_rs::{Parakeet, ExecutionConfig, ExecutionProvider};

let config = ExecutionConfig::new()
    .with_environment(parakeet_rs::ort::init().build()?)
    .with_execution_provider(ExecutionProvider::Cuda);
let mut parakeet = Parakeet::from_pretrained(".", Some(config))?;
```

Advanced session configuration via [ort SessionBuilder](https://docs.rs/ort/latest/ort/session/builder/struct.SessionBuilder.html):
```rust
let config = ExecutionConfig::new()
    .with_custom_configure(|builder| builder.with_memory_pattern(false));
```

If your app also uses `ort` directly, use the same environment for both: ONNX Runtime allows only one per process. `parakeet_rs::ort` is the `ort` version parakeet-rs is built with.

### Pure Rust backend (burn)

Every model also runs on [burn](https://burn.dev), without ONNX Runtime, on CPU or GPU. Same fp32 model files, same transcripts. Cohere's fp32 export is about 8.4 GB, all in memory. Details: [v0.4.0 release notes](https://github.com/altunenes/parakeet-rs/releases/tag/v0.4.0)
```toml
parakeet-rs = { version = "0.4", default-features = false, features = ["metal"] }  # "wgpu" for other GPUs, "burn" for CPU only
```

```rust
let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::BurnWgpu);
let mut parakeet = ParakeetTDT::from_pretrained("./tdt", Some(config))?;
```

Other GPU features: `vulkan`, `burn-cuda` and `burn-rocm`. For burn on the CPU, `apple-amx` (Apple Silicon) or `x86-v4` (AVX-512) speed up matrix multiplication. See `examples/burn.rs`.

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)
- CTC and TDT take up to about 4 to 5 minutes of audio. For longer files use a streaming model, Multitalker TDT, or split it
- Token timestamps: CTC, TDT

## License

Code: MIT OR Apache-2.0

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) by NVIDIA. This library does not distribute the models.
