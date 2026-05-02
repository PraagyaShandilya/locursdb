use std::num::ParseIntError;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum VectorIDError {
    #[error("Vec Dimensions mismatched: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("Duplicate Vec ID: {0}")]
    DuplicateId(String),

    #[error("Vector not found: {0}")]
    NotFound(String),
}

#[derive(Debug, thiserror::Error)]
pub enum DotEnvError {
    #[error(transparent)]
    Dotenv(#[from] dotenvy::Error),
    #[error("missing {key} in {path}")]
    MissingEnvVar { key: &'static str, path: PathBuf },
    #[error("invalid usize for {key}: {value}")]
    InvalidUsize {
        key: &'static str,
        value: String,
        #[source]
        source: ParseIntError,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error("OpenRouter returned {status}: {body}")]
    Api {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum TextError {
    #[error("failed to read {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to initialize sentence segmenter: {0}")]
    SegmenterInit(String),
}

#[derive(Debug, thiserror::Error)]
pub enum MainError {
    #[error(transparent)]
    TerminalIo(#[from] std::io::Error),
    #[error(transparent)]
    VectorIDError(#[from] VectorIDError),
    #[error(transparent)]
    DotEnvError(#[from] DotEnvError),
    #[error(transparent)]
    ApiError(#[from] ApiError),
    #[error(transparent)]
    TextError(#[from] TextError),
}
