use std::path::PathBuf;


#[derive(thiserror::Error,Debug)] 
pub enum VectorIDError{
    #[error("Vec Dimensions mismatched: expected {expected}, got {actual}")]
    DimMismatch{expected: usize, actual: usize},

    #[error("Duplicate Vec ID: {0}")]
    DuplicateId(String),
    
    #[error("Vector not found: {0}")]
    NotFound(String),
}

#[derive(Debug, thiserror::Error)]
pub enum MainError {
    #[error(transparent)]
    Dotenv(#[from] dotenvy::Error),
    #[error("missing {key} in {path}")]
    MissingEnvVar { key: &'static str, path: PathBuf },
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("OpenRouter returned {status}: {body}")]
    Api {
        status: reqwest::StatusCode,
        body: String,
    },
}