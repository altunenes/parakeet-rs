use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Ort(ort::Error),
    Audio(String),
    Model(String),
    Tokenizer(String),
    Config(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {e}"),
            Error::Ort(e) => write!(f, "ONNX Runtime error: {e}"),
            Error::Audio(msg) => write!(f, "Audio processing error: {msg}"),
            Error::Model(msg) => write!(f, "Model error: {msg}"),
            Error::Tokenizer(msg) => write!(f, "Tokenizer error: {msg}"),
            Error::Config(msg) => write!(f, "Config error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Ort(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Config(e.to_string())
    }
}

impl From<hound::Error> for Error {
    fn from(e: hound::Error) -> Self {
        Error::Audio(e.to_string())
    }
}
