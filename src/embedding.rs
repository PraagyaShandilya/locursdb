use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize)]
pub struct EmbeddingsRequest {
    input: Vec<String>,
    model: String,
    dimensions:  f32,
    encoding_format: String,
    input_type: String,
}

impl EmbeddingsRequest {
    pub fn new(input: Vec<String>, model: &str, dimensions: f32) -> Self {
        Self {
            input,
            model:model.to_string(),
            dimensions,
            encoding_format: "float".to_string(),
            input_type: "query".to_string(),
        }
    }

    pub fn new_fromstring(){

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
