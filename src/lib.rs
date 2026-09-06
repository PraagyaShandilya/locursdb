pub mod app;
pub mod application;
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
pub use embedding::{
    ApiClient, EmbeddingBatchProgress, EmbeddingsApiResponse, EmbeddingsRequest,
    EmbeddingsResponse, OpenRouterEmbedder, PseudoEmbedder,
};
pub use error::{ApiError, DotEnvError, MainError, TextError, VectorIDError};
pub use ingest::{FileType, Ingest};
pub use logging::{EmbeddingLogger, EmbeddingProgress};
pub use store_repository::StoreRepository;
pub use vector::{
    ChunkMetadata, ContentHash, DistanceMetric, DocumentId, Point, SourceUri, VectorID, VectorStore,
};
