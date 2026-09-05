use std::num::ParseIntError;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum VectorIDError {
    #[error("vector dimensions mismatched: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("vectors must contain at least one dimension")]
    EmptyVector,

    #[error("vector contains a non-finite value at index {index}")]
    NonFiniteValue { index: usize },

    #[error("cosine distance is undefined for a zero-norm vector")]
    ZeroNorm,

    #[error("collection length mismatch: {embeddings} embeddings for {inputs} inputs")]
    CollectionLengthMismatch { embeddings: usize, inputs: usize },

    #[error("duplicate vector ID: {0}")]
    DuplicateId(String),

    #[error("vector not found: {0}")]
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
    #[error("{key} must be greater than zero")]
    MustBePositive { key: &'static str },
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
    #[error("{field} must be greater than zero")]
    InvalidConfiguration { field: &'static str },
    #[error("embedding response count mismatch: expected {expected}, got {actual}")]
    EmbeddingCountMismatch { expected: usize, actual: usize },
    #[error("embedding {embedding_index} dimensions mismatched: expected {expected}, got {actual}")]
    EmbeddingDimMismatch {
        embedding_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("embedding {embedding_index} contains a non-finite value at index {value_index}")]
    NonFiniteEmbedding {
        embedding_index: usize,
        value_index: usize,
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

    #[error("chunk size must be greater than zero")]
    InvalidChunkSize,
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
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
