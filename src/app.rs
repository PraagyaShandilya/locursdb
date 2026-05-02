use base64::prelude::*;
use ulid::Ulid;

use crate::{
    ApiClient, AppConfig, ChunkMetadata, ContentHash, DistanceMetric, DocumentId, FileType, Ingest,
    MainError, Point, SourceUri, VectorID, VectorStore, tui,
};

pub async fn run() -> Result<(), MainError> {
    let config = AppConfig::load()?;
    tui::install_panic_hook();

    let api = ApiClient::new(
        config.dimensions,
        config.batch_size,
        config.embedding_concurrency,
        config.openrouter_api_key.clone(),
        config.model_name.clone(),
    );

    let Some(input) = tui::collect_input(config.corpus_path.clone())? else {
        return Ok(());
    };

    let corpus = Ingest::new(input.corpus_path, config.chunk_size, FileType::Txt);
    let inputs = corpus.chunks_from_file()?;
    let embeddings = api.convert_input_to_embeddings(inputs.clone()).await?;

    let mut store = VectorStore::new(DistanceMetric::Euclid);
    store.create_collections(embeddings, inputs, api.model_name().to_string())?;

    let results = run_query(&api, &store, input.query, config.top_k).await?;
    tui::show_results(results)?;

    Ok(())
}

async fn run_query(
    api: &ApiClient,
    store: &VectorStore,
    query: String,
    top_k: usize,
) -> Result<Vec<String>, MainError> {
    let query_embeddings = api.embeddings_api_call(vec![query.clone()]).await?;

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

    Ok(results)
}
