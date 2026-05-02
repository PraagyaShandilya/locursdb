use futures::{StreamExt, stream};

use crate::ApiError;

use super::{EmbeddingsApiResponse, EmbeddingsRequest};

#[derive(Debug, Clone)]
pub struct ApiClient {
    client: reqwest::Client,
    dimensions: usize,
    batch_size: usize,
    embedding_concurrency: usize,
    openrouter_api_key: String,
    model_name: String,
}

impl ApiClient {
    pub fn new(
        dimensions: usize,
        batch_size: usize,
        embedding_concurrency: usize,
        openrouter_api_key: String,
        model_name: String,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            dimensions,
            batch_size,
            embedding_concurrency,
            openrouter_api_key,
            model_name,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub async fn embeddings_api_call(
        &self,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        let request = EmbeddingsRequest::new(inputs, &self.model_name, self.dimensions);

        let res = self
            .client
            .post("https://openrouter.ai/api/v1/embeddings")
            .bearer_auth(&self.openrouter_api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = res.status();
        let body = res.text().await?;

        if !status.is_success() {
            return Err(ApiError::Api { status, body });
        }

        match serde_json::from_str::<EmbeddingsApiResponse>(&body)? {
            EmbeddingsApiResponse::Success(response) => Ok(response.into_embeddings()),
            EmbeddingsApiResponse::Error(error) => Err(ApiError::Api {
                status,
                body: format!("{} | body: {}", error.error.message, body),
            }),
        }
    }

    pub async fn convert_input_to_embeddings(
        &self,
        input: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        let batches: Vec<Vec<String>> = input
            .chunks(self.batch_size)
            .map(|batch| batch.to_vec())
            .collect();

        let batch_results: Vec<Result<Vec<Vec<f32>>, ApiError>> = stream::iter(batches)
            .map(|batch| self.embeddings_api_call(batch))
            .buffered(self.embedding_concurrency)
            .collect()
            .await;

        let mut embeddings = Vec::new();

        for result in batch_results {
            embeddings.extend(result?);
        }

        Ok(embeddings)
    }
}
