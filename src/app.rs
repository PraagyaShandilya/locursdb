use std::sync::Arc;

use crate::{
    ApiClient, AppConfig, DistanceMetric, EmbeddingLogger, MainError, OpenRouterEmbedder,
    application::{Application, ApplicationEvent, EphemeralSearchRequest, ProgressObserver},
    tui::{self, TuiInput},
};

/// Runs the interactive frontend.
pub async fn run() -> Result<(), MainError> {
    let config = AppConfig::load()?;
    tui::install_panic_hook();

    let logger = EmbeddingLogger::new("log")?;
    logger.trace("app started");
    logger.trace(format!("embedding log path: {}", logger.path().display()));

    let provider = OpenRouterEmbedder::new(ApiClient::new(
        config.dimensions,
        config.batch_size,
        config.embedding_concurrency,
        config.openrouter_api_key.clone(),
        config.model_name.clone(),
    ));
    let application = Application::from_environment(Arc::new(provider));

    let processing =
        tui::collect_input_and_process(config.corpus_path.clone(), |input, progress| async {
            logger.trace(format!(
                "selected corpus path: {}",
                input.corpus_path.display()
            ));
            logger.trace(format!(
                "query accepted: {} chars",
                input.query.chars().count()
            ));

            let event_logger = logger.clone();
            let observer = move |event: ApplicationEvent| {
                event_logger.trace(format!("application event: {event:?}"));
                let _ = progress.send(event);
            };
            let result = search_corpus(
                &application,
                input,
                config.chunk_size,
                config.dimensions,
                config.top_k,
                Some(&observer),
            )
            .await;
            match &result {
                Ok(results) => logger.trace(format!(
                    "ephemeral ingest and search completed: results={}",
                    results.len()
                )),
                Err(error) => logger.trace(format!("ephemeral ingest and search failed: {error}")),
            }
            result
        })
        .await?;

    let Some(results) = processing else {
        logger.trace("input cancelled; exiting");
        return Ok(());
    };
    tui::show_results(results)?;

    Ok(())
}

async fn search_corpus(
    application: &Application,
    input: TuiInput,
    chunk_size: usize,
    dimensions: usize,
    top_k: usize,
    observer: Option<&ProgressObserver>,
) -> Result<Vec<String>, MainError> {
    application
        .ingest_and_search(
            EphemeralSearchRequest {
                corpus_path: input.corpus_path,
                chunk_size,
                dimensions,
                metric: DistanceMetric::Euclid,
                query: input.query,
                top_k,
            },
            observer,
        )
        .await
        .map(|hits| {
            hits.into_iter()
                .map(|hit| hit.point.metadata.content)
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use super::*;
    use crate::PseudoEmbedder;

    #[tokio::test]
    async fn adapter_returns_chunk_contents_without_persisting() {
        let unique = format!("locursdb-tui-{}", ulid::Ulid::new());
        let root = std::env::temp_dir().join(unique);
        let corpus = root.join("corpus.txt");
        fs::create_dir_all(&root).unwrap();
        fs::write(&corpus, "alpha beta gamma delta").unwrap();
        let repository_root = root.join("stores");
        let application = Application::at_root(repository_root.clone(), Arc::new(PseudoEmbedder));

        let mut results = search_corpus(
            &application,
            TuiInput {
                corpus_path: corpus,
                query: "alpha".to_string(),
            },
            2,
            8,
            10,
            None,
        )
        .await
        .unwrap();
        results.sort();

        assert_eq!(results, ["alpha beta", "gamma delta"]);
        assert!(!repository_root.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
