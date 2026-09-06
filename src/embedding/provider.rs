//! Concrete adapters for the application's embedding port.

use crate::{
    ApiClient,
    application::{ApplicationEvent, EmbedFuture, Embedder, ProgressObserver},
};

#[derive(Debug, Clone, Default)]
pub struct PseudoEmbedder;

impl Embedder for PseudoEmbedder {
    fn embed<'a>(
        &'a self,
        inputs: Vec<String>,
        dimensions: usize,
        observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a> {
        Box::pin(async move {
            let total_batches = usize::from(!inputs.is_empty());
            report(observer, 0, total_batches);
            let embeddings = super::local::embed_batch(&inputs, dimensions);
            if total_batches != 0 {
                report(observer, 1, 1);
            }
            Ok(embeddings)
        })
    }
}

#[derive(Debug, Clone)]
pub struct OpenRouterEmbedder {
    client: ApiClient,
}

impl OpenRouterEmbedder {
    pub fn new(client: ApiClient) -> Self {
        Self { client }
    }
}

impl Embedder for OpenRouterEmbedder {
    fn embed<'a>(
        &'a self,
        inputs: Vec<String>,
        dimensions: usize,
        observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a> {
        Box::pin(async move {
            self.client
                .with_dimensions(dimensions)
                .convert_input_to_embeddings_with_callback(inputs, |progress| {
                    report(observer, progress.completed_batches, progress.total_batches);
                })
                .await
                .map_err(Into::into)
        })
    }
}

fn report(observer: Option<&ProgressObserver>, completed_batches: usize, total_batches: usize) {
    if let Some(observer) = observer {
        observer(ApplicationEvent::EmbeddingProgress {
            completed_batches,
            total_batches,
        });
    }
}
