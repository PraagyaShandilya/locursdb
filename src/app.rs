use base64::prelude::*;
use ulid::Ulid;

use crate::{
    ApiClient, AppConfig, ChunkMetadata, ContentHash, DistanceMetric, DocumentId, FileType,
    Ingest, MainError, Point, SourceUri, VectorID, VectorStore,
};

pub async fn run() -> Result<(), MainError> {
    let config = AppConfig::load()?;

    let api = ApiClient::new(
        config.dimensions,
        config.batch_size,
        config.openrouter_api_key.clone(),
        config.model_name.clone(),
    );

    let corpus = Ingest::new(config.corpus_path.clone(), config.chunk_size, FileType::Txt);
    let inputs = corpus.chunks_from_file()?;
    let embeddings = api.convert_input_to_embeddings(inputs.clone()).await?;

    let mut store = VectorStore::new(DistanceMetric::Euclid);
    store.create_collections(embeddings, inputs, api.model_name().to_string())?;

    run_query_demo(&api, &store).await?;

    Ok(())
}

async fn run_query_demo(api: &ApiClient, store: &VectorStore) -> Result<(), MainError> {
    let query = "moses talked aboue mount sinai".to_string();
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

    for point in store.get_top_k(&query_point, 5) {
        let decoded_text = match BASE64_STANDARD.decode(point.metadata.content_hash.0.as_bytes()) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(err) => format!("<base64 decode error: {err}>"),
        };
        println!("{}", decoded_text);
    }

    Ok(())
}
