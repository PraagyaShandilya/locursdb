mod client;
pub(crate) mod local;
mod types;

pub use client::ApiClient;
pub use types::{EmbeddingsApiResponse, EmbeddingsRequest, EmbeddingsResponse};
