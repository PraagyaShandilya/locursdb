mod client;
pub(crate) mod local;
mod provider;
mod types;

pub use client::{ApiClient, EmbeddingBatchProgress};
pub use provider::{OpenRouterEmbedder, PseudoEmbedder};
pub use types::{EmbeddingsApiResponse, EmbeddingsRequest, EmbeddingsResponse};
