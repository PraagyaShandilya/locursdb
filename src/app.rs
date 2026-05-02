use base64::prelude::*;
use ulid::Ulid;

use crate::{
    ApiClient, AppConfig, ChunkMetadata, ContentHash, DistanceMetric, DocumentId,
    EmbeddingProgress, FileType, Ingest, MainError, Point, SourceUri, VectorID, VectorStore, tui,
};

pub async fn run() -> Result<(), MainError> {
    let config = AppConfig::load()?;
    tui::install_panic_hook();

    let logger = crate::EmbeddingLogger::new("log")?;
    logger.trace("app started");
    logger.trace(format!("embedding log path: {}", logger.path().display()));

    let api = ApiClient::new(
        config.dimensions,
        config.batch_size,
        config.embedding_concurrency,
        config.openrouter_api_key.clone(),
        config.model_name.clone(),
    );

    let processing =
        tui::collect_input_and_process(config.corpus_path.clone(), |input, progress| {
            let api = api.clone();
            let logger = logger.clone();
            let model_name = api.model_name().to_string();
            let chunk_size = config.chunk_size;
            let top_k = config.top_k;

            async move {
                logger.trace(format!(
                    "selected corpus path: {}",
                    input.corpus_path.display()
                ));
                logger.trace(format!(
                    "query accepted: {} chars",
                    input.query.chars().count()
                ));
                let _ = progress.send(EmbeddingProgress::new(0, 0, "reading and chunking corpus"));

                let corpus = Ingest::new(input.corpus_path, chunk_size, FileType::Txt);
                logger.trace("reading and chunking corpus started");
                let inputs = corpus.chunks_from_file()?;
                logger.trace(format!(
                    "reading and chunking corpus finished: chunks={}",
                    inputs.len()
                ));

                let embeddings = api
                    .convert_input_to_embeddings_with_progress(
                        inputs.clone(),
                        Some(&logger),
                        Some(progress.clone()),
                    )
                    .await?;

                let _ = progress.send(EmbeddingProgress::new(0, 0, "creating vector store"));
                logger.trace("vector store creation started");
                let mut store = VectorStore::new(DistanceMetric::Euclid);
                store.create_collections(embeddings, inputs, model_name)?;
                logger.trace("vector store creation finished");

                let _ = progress.send(EmbeddingProgress::new(
                    0,
                    0,
                    "embedding query and searching",
                ));
                run_query(&api, &store, input.query, top_k, &logger).await
            }
        })
        .await?;

    let Some(results) = processing else {
        logger.trace("input cancelled; exiting");
        return Ok(());
    };
    tui::show_results(results)?;

    Ok(())
}

async fn run_query(
    api: &ApiClient,
    store: &VectorStore,
    query: String,
    top_k: usize,
    logger: &crate::EmbeddingLogger,
) -> Result<Vec<String>, MainError> {
    logger.trace(format!("query embedding started: top_k={top_k}"));
    let query_embeddings = api.embeddings_api_call(vec![query.clone()]).await?;
    logger.trace("query embedding finished");

    let query_point = Point {
        id: VectorID::new(),
        vec: query_embeddings[0].clone(),
        metadata: ChunkMetadata {
            document_id: DocumentId(Ulid::new().to_string()),
            source_uri: SourceUri(api.model_name().to_string()),
            chunk_index: 0,
            content_hash: ContentHash(BASE64_STANDARD.encode(query.as_bytes())),
        },
    };

    logger.trace("top-k search started");
    let results = store
        .get_top_k(&query_point, top_k)
        .into_iter()
        .map(
            |point| match BASE64_STANDARD.decode(point.metadata.content_hash.0.as_bytes()) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(err) => format!("<base64 decode error: {err}>"),
            },
        )
        .collect();
    logger.trace("top-k search finished");

    Ok(results)
}
