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
//! use parakeet_rs::{Parakeet, Transcriber, TimestampMode};
//!
//! // Load the model
//! let mut parakeet = Parakeet::from_pretrained(".")?;
//!
//! // Transcribe audio samples (see examples/raw.rs for audio loading)
//! let result = parakeet.transcribe_samples(audio, sample_rate, channels, Some(TimestampMode::Words))?;
//! println!("Transcription: {}", result.text);
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
mod decoder_tdt;
mod error;
mod execution;
mod model;
mod model_eou;
mod model_nemotron;
mod model_tdt;
mod nemotron;
mod parakeet;
mod parakeet_eou;
mod parakeet_tdt;
#[cfg(feature = "sortformer")]
pub mod sortformer;
mod timestamps;
mod transcriber;
mod vocab;

pub use error::{Error, Result};
pub use execution::{ExecutionProvider, ModelConfig as ExecutionConfig};
pub use parakeet::Parakeet;
pub use parakeet_tdt::ParakeetTDT;
pub use timestamps::TimestampMode;
pub use transcriber::*;

pub use config::{ModelConfig as ModelConfigJson, PreprocessorConfig};

pub use decoder::{ParakeetDecoder, TimedToken, TranscriptionResult};
pub use model::ParakeetModel;
pub use model_eou::ParakeetEOUModel;
pub use model_nemotron::{NemotronEncoderCache, NemotronModel, NemotronModelConfig};
pub use nemotron::{Nemotron, SentencePieceVocab};
pub use parakeet_eou::ParakeetEOU;
