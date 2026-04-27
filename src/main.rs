use std::fs;
use std::path::{Path, PathBuf};
use pragmatic_segmenter::Segmenter;
use ulid::Ulid;


use locursdb::{
    ApiError,
    ChunkMetadata, 
    ContentHash, 
    DistanceMetric, 
    DocumentId,
    DotEnvError,
    EmbeddingsApiResponse,
    EmbeddingsRequest, 
    MainError, 
    SourceUri,
    TextError,
    VectorID
    ,
    VectorStore, 
};




fn chunks_from_file(path: &Path, chunk_size: usize) -> Result<Vec<String>, TextError> {
    
    let text = fs::read_to_string(path).map_err(|source| TextError::Read {
        path: path.to_path_buf(),
        source,
    })?;

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let n = words.len();


    for i in (0..n).step_by(chunk_size){
        let end = (i + chunk_size).min(words.len());
        let chunk = words[i .. end].join("");
        chunks.push(chunk);
    }
        

    Ok(chunks)
}



fn load_openrouter_api_key() -> Result<String, DotEnvError> {
    let env_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env");
    let iter = dotenvy::from_path_iter(&env_path)?;

    for item in iter {
        let (key, value) = item?;
        if key == "OPENROUTER_API_KEY" {
            return Ok(value);
        }
    }

    Err(DotEnvError::MissingEnvVar {
        key: "OPENROUTER_API_KEY",
        path: env_path,
    })
}


async fn embeddings_api_call(
    inputs: Vec<String>,
    dimensions: usize,
    openrouter_api_key: &str,
    model_name: &str,
) -> Result<Vec<Vec<f32>>, ApiError> {
    let request = EmbeddingsRequest::new(
        inputs,
        model_name,
        dimensions
    );

    
    let client = reqwest::Client::new();
    let res = client
        .post("https://openrouter.ai/api/v1/embeddings")
        .bearer_auth(openrouter_api_key)
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



#[tokio::main]
async fn main() -> Result<(), MainError> {
    const BATCH_SIZE: usize = 128;
    const DIMENSIONS: usize = 512;
    const MODEL_NAME: &str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"; 

    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/sample.txt");
    
    let inputs = chunks_from_file(&corpus_path, BATCH_SIZE)?;
    

    let openrouter_api_key = load_openrouter_api_key()?;
    let mut embeddings = Vec::new();

    for batch in inputs.chunks(BATCH_SIZE) {
        embeddings.extend(
            embeddings_api_call(
                batch.to_vec(),
                DIMENSIONS,
                &openrouter_api_key,
                MODEL_NAME,
            )
            .await?,
        );
    }

    let mut store: VectorStore = VectorStore::new(DistanceMetric::Euclid);
    println!("{:?}", embeddings.len());

    for (idx, (embed, input)) in embeddings.into_iter().zip(inputs).enumerate() {
        let meta: ChunkMetadata = ChunkMetadata {
            document_id: DocumentId(Ulid::new().to_string()),
            source_uri: SourceUri(MODEL_NAME.to_string()),
            chunk_index: idx,
            content_hash: ContentHash(blake3::hash(input.as_bytes()).to_hex().to_string()),
        };

        store.upsert(VectorID::new(), embed, meta)?
    }


    Ok(())
}
