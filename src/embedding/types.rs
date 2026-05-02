use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct EmbeddingsRequest {
    input: Vec<String>,
    model: String,
    dimensions: usize,
    encoding_format: String,
}

impl EmbeddingsRequest {
    pub fn new(input: Vec<String>, model: &str, dimensions: usize) -> Self {
        Self {
            input,
            model: model.to_string(),
            dimensions,
            encoding_format: "float".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

impl EmbeddingsResponse {
    pub fn into_embeddings(self) -> Vec<Vec<f32>> {
        self.data.into_iter().map(|item| item.embedding).collect()
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsApiResponse {
    Success(EmbeddingsResponse),
    Error(OpenRouterErrorResponse),
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterErrorResponse {
    pub error: OpenRouterError,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterError {
    pub message: String,
}
