use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures::{StreamExt, stream};
use tokio::sync::mpsc::UnboundedSender;

use crate::{ApiError, EmbeddingLogger, EmbeddingProgress};

use super::{EmbeddingsApiResponse, EmbeddingsRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingBatchProgress {
    pub completed_batches: usize,
    pub total_batches: usize,
}

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

    /// Clones the provider configuration while reusing reqwest's pooled client.
    pub fn with_dimensions(&self, dimensions: usize) -> Self {
        let mut client = self.clone();
        client.dimensions = dimensions;
        client
    }

    fn validate_configuration(&self) -> Result<(), ApiError> {
        for (field, value) in [
            ("dimensions", self.dimensions),
            ("batch size", self.batch_size),
            ("embedding concurrency", self.embedding_concurrency),
        ] {
            if value == 0 {
                return Err(ApiError::InvalidConfiguration { field });
            }
        }
        Ok(())
    }

    fn validate_embeddings(
        &self,
        expected_count: usize,
        embeddings: &[Vec<f32>],
    ) -> Result<(), ApiError> {
        if embeddings.len() != expected_count {
            return Err(ApiError::EmbeddingCountMismatch {
                expected: expected_count,
                actual: embeddings.len(),
            });
        }

        for (embedding_index, embedding) in embeddings.iter().enumerate() {
            if embedding.len() != self.dimensions {
                return Err(ApiError::EmbeddingDimMismatch {
                    embedding_index,
                    expected: self.dimensions,
                    actual: embedding.len(),
                });
            }
            if let Some(value_index) = embedding.iter().position(|value| !value.is_finite()) {
                return Err(ApiError::NonFiniteEmbedding {
                    embedding_index,
                    value_index,
                });
            }
        }

        Ok(())
    }

    pub async fn embeddings_api_call(
        &self,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        self.validate_configuration()?;
        let expected_count = inputs.len();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
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

        let embeddings = match serde_json::from_str::<EmbeddingsApiResponse>(&body)? {
            EmbeddingsApiResponse::Success(response) => response.into_embeddings(),
            EmbeddingsApiResponse::Error(error) => {
                return Err(ApiError::Api {
                    status,
                    body: format!("{} | body: {}", error.error.message, body),
                });
            }
        };
        self.validate_embeddings(expected_count, &embeddings)?;
        Ok(embeddings)
    }

    pub async fn convert_input_to_embeddings(
        &self,
        input: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        self.convert_input_to_embeddings_with_logger(input, None)
            .await
    }

    pub async fn convert_input_to_embeddings_with_logger(
        &self,
        input: Vec<String>,
        logger: Option<&EmbeddingLogger>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        self.convert_input_to_embeddings_with_progress(input, logger, None)
            .await
    }

    pub async fn convert_input_to_embeddings_with_progress(
        &self,
        input: Vec<String>,
        logger: Option<&EmbeddingLogger>,
        progress: Option<UnboundedSender<EmbeddingProgress>>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        self.convert_input_to_embeddings_inner(input, logger, progress, None)
            .await
    }

    /// Converts inputs while reporting provider-neutral batch counts.
    ///
    /// The channel-based progress API remains available for existing callers.
    pub async fn convert_input_to_embeddings_with_callback<F>(
        &self,
        input: Vec<String>,
        callback: F,
    ) -> Result<Vec<Vec<f32>>, ApiError>
    where
        F: Fn(EmbeddingBatchProgress) + Send + Sync,
    {
        self.convert_input_to_embeddings_inner(input, None, None, Some(&callback))
            .await
    }

    async fn convert_input_to_embeddings_inner(
        &self,
        input: Vec<String>,
        logger: Option<&EmbeddingLogger>,
        progress: Option<UnboundedSender<EmbeddingProgress>>,
        callback: Option<&(dyn Fn(EmbeddingBatchProgress) + Send + Sync)>,
    ) -> Result<Vec<Vec<f32>>, ApiError> {
        self.validate_configuration()?;
        let batches: Vec<Vec<String>> = input
            .chunks(self.batch_size)
            .map(|batch| batch.to_vec())
            .collect();
        let batch_count = batches.len();

        if let Some(logger) = logger {
            logger.trace(format!(
                "embedding conversion started: chunks={}, batch_size={}, batches={}, concurrency={}",
                input.len(), self.batch_size, batch_count, self.embedding_concurrency
            ));
        }
        if let Some(progress) = &progress {
            let _ = progress.send(EmbeddingProgress::new(
                0,
                batch_count,
                format!(
                    "starting embedding: {} chunks in {batch_count} batches",
                    input.len()
                ),
            ));
        }
        if let Some(callback) = callback {
            callback(EmbeddingBatchProgress {
                completed_batches: 0,
                total_batches: batch_count,
            });
        }

        let completed_batches = Arc::new(AtomicUsize::new(0));

        let batch_results: Vec<Result<Vec<Vec<f32>>, ApiError>> = stream::iter(
            batches
                .into_iter()
                .enumerate()
                .map(|(index, batch)| (index + 1, batch)),
        )
        .map(|(batch_number, batch)| {
            let logger = logger.cloned();
            let progress = progress.clone();
            let completed_batches = completed_batches.clone();
            async move {
                if let Some(logger) = &logger {
                    logger.trace(format!(
                        "embedding batch {batch_number}/{batch_count} started: inputs={}",
                        batch.len()
                    ));
                }

                let result = self.embeddings_api_call(batch).await;

                if let Some(logger) = &logger {
                    match &result {
                        Ok(embeddings) => logger.trace(format!(
                            "embedding batch {batch_number}/{batch_count} finished: embeddings={}",
                            embeddings.len()
                        )),
                        Err(error) => logger.trace(format!(
                            "embedding batch {batch_number}/{batch_count} failed: {error}"
                        )),
                    }
                }
                let completed = completed_batches.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(progress) = &progress {
                    let message = match &result {
                        Ok(embeddings) => format!(
                            "finished batch {batch_number}/{batch_count}: {} embeddings",
                            embeddings.len()
                        ),
                        Err(error) => format!("failed batch {batch_number}/{batch_count}: {error}"),
                    };
                    let _ = progress.send(EmbeddingProgress::new(completed, batch_count, message));
                }
                if let Some(callback) = callback {
                    callback(EmbeddingBatchProgress {
                        completed_batches: completed,
                        total_batches: batch_count,
                    });
                }

                result
            }
        })
        .buffered(self.embedding_concurrency)
        .collect()
        .await;

        let mut embeddings = Vec::new();

        for result in batch_results {
            embeddings.extend(result?);
        }

        if let Some(logger) = logger {
            logger.trace(format!(
                "embedding conversion finished: embeddings={}",
                embeddings.len()
            ));
        }
        if let Some(progress) = &progress {
            let _ = progress.send(EmbeddingProgress::new(
                batch_count,
                batch_count,
                format!("embedding finished: {} embeddings", embeddings.len()),
            ));
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::ApiClient;
    use crate::ApiError;

    fn client(dimensions: usize, batch_size: usize, concurrency: usize) -> ApiClient {
        ApiClient::new(
            dimensions,
            batch_size,
            concurrency,
            "test-key".to_string(),
            "test-model".to_string(),
        )
    }

    #[test]
    fn dimensions_builder_changes_only_requested_dimensions() {
        let original = client(2, 3, 4);
        let resized = original.with_dimensions(8);

        assert_eq!(original.dimensions, 2);
        assert_eq!(resized.dimensions, 8);
        assert_eq!(resized.batch_size, 3);
        assert_eq!(resized.embedding_concurrency, 4);
        assert_eq!(resized.model_name(), "test-model");
    }

    #[test]
    fn configuration_rejects_zero_values() {
        let error = client(4, 0, 1).validate_configuration().unwrap_err();

        assert!(matches!(
            error,
            ApiError::InvalidConfiguration {
                field: "batch size"
            }
        ));
    }

    #[test]
    fn embedding_validation_rejects_wrong_count() {
        let error = client(2, 1, 1)
            .validate_embeddings(2, &[vec![1.0, 2.0]])
            .unwrap_err();

        assert!(matches!(
            error,
            ApiError::EmbeddingCountMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn embedding_validation_rejects_wrong_dimensions() {
        let error = client(3, 1, 1)
            .validate_embeddings(1, &[vec![1.0, 2.0]])
            .unwrap_err();

        assert!(matches!(
            error,
            ApiError::EmbeddingDimMismatch {
                embedding_index: 0,
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn embedding_validation_rejects_non_finite_values() {
        let error = client(2, 1, 1)
            .validate_embeddings(1, &[vec![1.0, f32::NAN]])
            .unwrap_err();

        assert!(matches!(
            error,
            ApiError::NonFiniteEmbedding {
                embedding_index: 0,
                value_index: 1
            }
        ));
    }
}
