use crate::{ApiError, EmbeddingsApiResponse, EmbeddingsRequest};

#[derive(Debug, Clone)]
pub struct ApiClient {
    dimensions: usize,
    batch_size: usize,
    openrouter_api_key: String,
    model_name: String,
}

impl ApiClient {
    pub fn new(
        dimensions: usize,
        batch_size: usize,
        openrouter_api_key: String,
        model_name: String,
    ) -> Self {
        Self {
            dimensions,
            batch_size,
            openrouter_api_key,
            model_name,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub async fn embeddings_api_call(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, ApiError> {
        let request = EmbeddingsRequest::new(inputs, &self.model_name, self.dimensions);

        let client = reqwest::Client::new();
        let res = client
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
        let mut embeddings = Vec::new();

        for batch in input.chunks(self.batch_size) {
            embeddings.extend(self.embeddings_api_call(batch.to_vec()).await?);
        }

        Ok(embeddings)
    }
}
