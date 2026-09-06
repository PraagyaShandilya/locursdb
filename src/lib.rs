pub mod app;
mod cli;
mod config;
mod embedding;
mod error;
mod ingest;
mod logging;
mod store_repository;
mod tui;
mod vector;

pub use app::run;
pub use cli::run_cli;
pub use config::AppConfig;
pub use embedding::{ApiClient, EmbeddingsApiResponse, EmbeddingsRequest, EmbeddingsResponse};
pub use error::{ApiError, DotEnvError, MainError, TextError, VectorIDError};
pub use ingest::{FileType, Ingest};
pub use logging::{EmbeddingLogger, EmbeddingProgress};
pub use vector::{
    ChunkMetadata, ContentHash, DistanceMetric, DocumentId, Point, SourceUri, VectorID, VectorStore,
};
