//! # parakeet-rs
//!
//! Rust bindings for NVIDIA's Parakeet speech recognition model using ONNX Runtime.
//!
//! Parakeet is a state-of-the-art automatic speech recognition (ASR) model developed by NVIDIA,
//! based on the FastConformer-TDT architecture with 600 million parameters.
//!
//! ## Features
//!
//! - Easy-to-use API for speech-to-text transcription
//! - Support for ONNX format models
//! - 16kHz mono audio input
//! - Punctuation and capitalization included in output
//! - Fast inference using ONNX Runtime
//!
//! ## Quick Start
//!
//! ```ignore
//! use parakeet_rs::Parakeet;
//!
//! // Load the model
//! let parakeet = Parakeet::from_pretrained(".")?;
//!
//! // Transcribe audio file
//! let text = parakeet.transcribe("audio.wav")?;
//! println!("Transcription: {}", text);
//! ```
//!
//! ## Model Requirements
//!
//! Your model directory should contain:
//! - `model.onnx` - The ONNX model file
//! - `model.onnx_data` - External model weights
//! - `config.json` - Model configuration
//! - `preprocessor_config.json` - Audio preprocessing configuration
//! - `tokenizer.json` - Tokenizer vocabulary
//! - `tokenizer_config.json` - Tokenizer configuration
//!
//! ## Audio Requirements
//!
//! - Format: WAV
//! - Sample Rate: 16kHz
//! - Channels: Mono (stereo will be converted automatically)
//! - Bit Depth: 16-bit PCM or 32-bit float

mod audio;
mod config;
mod decoder;
mod error;
mod execution;
mod model;
mod parakeet;

pub use error::{Error, Result};
pub use execution::{ExecutionProvider, ModelConfig as ExecutionConfig};
pub use parakeet::Parakeet;

pub use config::{ModelConfig as ModelConfigJson, PreprocessorConfig};

pub use model::ParakeetModel;
pub use decoder::{ParakeetDecoder, TimedToken, TranscriptionResult};
