//! In-process application workflows shared by command-line and interactive adapters.

use std::{collections::HashMap, future::Future, path::PathBuf, pin::Pin, sync::Arc};

use serde::Serialize;
use ulid::Ulid;

use crate::{
    ApiError, ChunkMetadata, ContentHash, DistanceMetric, DocumentId, FileType, Ingest, MainError,
    Point, SourceUri, VectorID, VectorStore, ingest::source, store_repository::StoreRepository,
};

pub type EmbedFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, MainError>> + Send + 'a>>;
/// Called synchronously during a workflow. Observers should be quick and non-blocking.
/// Stages indicate work starting; the operation's Result signals completion or failure.
pub type ProgressObserver = dyn Fn(ApplicationEvent) + Send + Sync;

/// An embedding provider. Implementations may be shared by multiple workflows.
pub trait Embedder: Send + Sync {
    fn embed<'a>(
        &'a self,
        inputs: Vec<String>,
        dimensions: usize,
        observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowStage {
    DiscoveringSources,
    Chunking,
    Embedding,
    Persisting,
    Searching,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplicationEvent {
    Stage(WorkflowStage),
    SourcesDiscovered {
        files: usize,
    },
    FileSkipped {
        path: PathBuf,
    },
    /// Provider-neutral completed/total batch counts.
    EmbeddingProgress {
        completed_batches: usize,
        total_batches: usize,
    },
}

/// Typed workflow boundary. UI adapters supply requests and present the returned values.
pub struct Application {
    repository: StoreRepository,
    embedder: Arc<dyn Embedder>,
}

impl Application {
    pub fn new(repository: StoreRepository, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            repository,
            embedder,
        }
    }

    pub fn at_root(root: PathBuf, embedder: Arc<dyn Embedder>) -> Self {
        Self::new(StoreRepository::new(root), embedder)
    }

    pub fn from_environment(embedder: Arc<dyn Embedder>) -> Self {
        Self::new(StoreRepository::from_environment(), embedder)
    }

    pub fn create(&self, request: CreateStoreRequest) -> Result<CreatedStore, MainError> {
        require_positive(request.dimensions, "dimensions")?;
        let created =
            self.repository
                .create(request.name.clone(), request.metric, request.dimensions)?;
        Ok(CreatedStore {
            name: request.name,
            metric: created.config.metric,
            dimensions: request.dimensions,
            path: created.path,
        })
    }

    pub async fn add_content(
        &self,
        request: AddContentRequest,
        observer: Option<&ProgressObserver>,
    ) -> Result<AddContentResult, MainError> {
        let mut open = self.repository.open(&request.name)?;
        let vector = self
            .vector_or_embed(
                request.vector,
                vec![request.content.clone()],
                open.config.dimensions,
                observer,
            )
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| invalid_error("embedding response was empty"))?;
        let id = request.id.unwrap_or_default();
        let metadata = content_metadata(
            request
                .document_id
                .unwrap_or_else(|| DocumentId(Ulid::new().to_string())),
            request
                .source_uri
                .unwrap_or_else(|| SourceUri("content://add".to_string())),
            request.chunk_index.unwrap_or(open.store.len()),
            request.content,
            request.labels,
            None,
            None,
            None,
            None,
            request.session_folder,
        );
        open.store.upsert(id, vector, metadata)?;
        emit(observer, ApplicationEvent::Stage(WorkflowStage::Persisting));
        self.repository.save(&open)?;
        Ok(AddContentResult {
            id,
            points: open.store.len(),
        })
    }

    /// Stages every file in memory and saves only after all embeddings validate.
    /// This does not make the repository's final disk write crash-atomic.
    pub async fn add_path(
        &self,
        request: AddPathRequest,
        observer: Option<&ProgressObserver>,
    ) -> Result<AddPathResult, MainError> {
        if request.path.as_os_str().is_empty() {
            return invalid("path must not be empty");
        }
        require_positive(request.chunk_size, "chunk size")?;
        if request.chunk_overlap >= request.chunk_size {
            return invalid("chunk overlap must be smaller than chunk size");
        }

        let mut open = self.repository.open(&request.name)?;
        emit(
            observer,
            ApplicationEvent::Stage(WorkflowStage::DiscoveringSources),
        );
        let files = source::discover(&request.path)?;
        emit(
            observer,
            ApplicationEvent::SourcesDiscovered { files: files.len() },
        );
        let mut added = 0;
        let mut skipped = 0;
        for file in files {
            emit(observer, ApplicationEvent::Stage(WorkflowStage::Chunking));
            let chunks = match source::read_chunks(&file, request.chunk_size, request.chunk_overlap)
            {
                Ok(chunks) => chunks,
                Err(_) => {
                    skipped += 1;
                    emit(observer, ApplicationEvent::FileSkipped { path: file });
                    continue;
                }
            };
            let inputs = chunks
                .iter()
                .map(|chunk| chunk.content.clone())
                .collect::<Vec<_>>();
            let vectors = self.embed(inputs, open.config.dimensions, observer).await?;
            let language = source::language(&file).map(str::to_string);
            let path = file.to_string_lossy().into_owned();
            let document_id = DocumentId(Ulid::new().to_string());
            for (chunk_index, (chunk, vector)) in chunks.into_iter().zip(vectors).enumerate() {
                let mut labels = request.labels.clone();
                labels.insert("path".to_string(), path.clone());
                labels.insert("start_line".to_string(), chunk.start_line.to_string());
                labels.insert("end_line".to_string(), chunk.end_line.to_string());
                if let Some(language) = &language {
                    labels.insert("language".to_string(), language.clone());
                }
                let metadata = content_metadata(
                    document_id.clone(),
                    SourceUri(path.clone()),
                    chunk_index,
                    chunk.content,
                    labels,
                    Some(path.clone()),
                    Some(chunk.start_line),
                    Some(chunk.end_line),
                    language.clone(),
                    request.session_folder.clone(),
                );
                open.store.upsert(VectorID::new(), vector, metadata)?;
                added += 1;
            }
        }
        emit(observer, ApplicationEvent::Stage(WorkflowStage::Persisting));
        self.repository.save(&open)?;
        Ok(AddPathResult {
            chunks_added: added,
            files_skipped: skipped,
            points: open.store.len(),
        })
    }

    pub async fn search(
        &self,
        request: SearchRequest,
        observer: Option<&ProgressObserver>,
    ) -> Result<Vec<SearchHit>, MainError> {
        let open = self.repository.open(&request.name)?;
        let vector = self
            .vector_or_embed(
                request.vector,
                vec![request.content],
                open.config.dimensions,
                observer,
            )
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| invalid_error("embedding response was empty"))?;
        emit(observer, ApplicationEvent::Stage(WorkflowStage::Searching));
        search_store(&open.store, vector, request.top_k, &request.filters)
    }

    pub fn delete(&self, request: DeleteRequest) -> Result<DeleteResult, MainError> {
        match request.target {
            DeleteTarget::Store => {
                self.repository.delete(&request.name)?;
                Ok(DeleteResult {
                    id: None,
                    points: None,
                })
            }
            DeleteTarget::Point(id) => {
                let mut open = self.repository.open(&request.name)?;
                open.store.delete(id);
                self.repository.save(&open)?;
                Ok(DeleteResult {
                    id: Some(id),
                    points: Some(open.store.len()),
                })
            }
        }
    }

    pub fn list(&self) -> Result<Vec<StoreSummary>, MainError> {
        Ok(self
            .repository
            .list()?
            .into_iter()
            .map(|store| StoreSummary {
                name: store.name,
                metric: store.metric,
                dimensions: store.dimensions,
                points: store.points,
                path: store.path,
            })
            .collect())
    }

    /// Builds an in-memory store from whitespace-delimited word chunks and searches it.
    pub async fn ingest_and_search(
        &self,
        request: EphemeralSearchRequest,
        observer: Option<&ProgressObserver>,
    ) -> Result<Vec<SearchHit>, MainError> {
        if request.corpus_path.as_os_str().is_empty() {
            return invalid("corpus path must not be empty");
        }
        require_positive(request.chunk_size, "chunk size")?;
        require_positive(request.dimensions, "dimensions")?;

        emit(observer, ApplicationEvent::Stage(WorkflowStage::Chunking));
        let inputs = Ingest::new(
            request.corpus_path.clone(),
            request.chunk_size,
            FileType::Txt,
        )
        .chunks_from_file()?;
        let embeddings = self
            .embed(inputs.clone(), request.dimensions, observer)
            .await?;
        let query_embeddings = self
            .embed(vec![request.query], request.dimensions, observer)
            .await?;
        let query_vector = query_embeddings
            .into_iter()
            .next()
            .ok_or_else(|| invalid_error("embedding response was empty"))?;

        let mut store = VectorStore::with_dimensions(request.metric, request.dimensions);
        let document_id = DocumentId(Ulid::new().to_string());
        let path = request.corpus_path.to_string_lossy().into_owned();
        for (chunk_index, (content, vector)) in inputs.into_iter().zip(embeddings).enumerate() {
            let metadata = content_metadata(
                document_id.clone(),
                SourceUri(path.clone()),
                chunk_index,
                content,
                HashMap::new(),
                Some(path.clone()),
                None,
                None,
                source::language(&request.corpus_path).map(str::to_string),
                None,
            );
            store.upsert(VectorID::new(), vector, metadata)?;
        }
        emit(observer, ApplicationEvent::Stage(WorkflowStage::Searching));
        search_store(&store, query_vector, request.top_k, &HashMap::new())
    }

    async fn vector_or_embed(
        &self,
        vector: Option<Vec<f32>>,
        inputs: Vec<String>,
        dimensions: usize,
        observer: Option<&ProgressObserver>,
    ) -> Result<Vec<Vec<f32>>, MainError> {
        if let Some(vector) = vector {
            validate_embeddings(inputs.len(), dimensions, std::slice::from_ref(&vector))?;
            Ok(vec![vector])
        } else {
            self.embed(inputs, dimensions, observer).await
        }
    }

    async fn embed(
        &self,
        inputs: Vec<String>,
        dimensions: usize,
        observer: Option<&ProgressObserver>,
    ) -> Result<Vec<Vec<f32>>, MainError> {
        require_positive(dimensions, "dimensions")?;
        emit(observer, ApplicationEvent::Stage(WorkflowStage::Embedding));
        let expected = inputs.len();
        let embeddings = self.embedder.embed(inputs, dimensions, observer).await?;
        validate_embeddings(expected, dimensions, &embeddings)?;
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
pub struct CreateStoreRequest {
    pub name: String,
    pub metric: DistanceMetric,
    pub dimensions: usize,
}
#[derive(Debug, Clone, Serialize)]
pub struct CreatedStore {
    pub name: String,
    pub metric: DistanceMetric,
    pub dimensions: usize,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct AddContentRequest {
    pub name: String,
    pub content: String,
    pub vector: Option<Vec<f32>>,
    pub id: Option<VectorID>,
    pub document_id: Option<DocumentId>,
    pub source_uri: Option<SourceUri>,
    pub chunk_index: Option<usize>,
    pub labels: HashMap<String, String>,
    pub session_folder: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct AddContentResult {
    pub id: VectorID,
    pub points: usize,
}

#[derive(Debug, Clone)]
pub struct AddPathRequest {
    pub name: String,
    pub path: PathBuf,
    pub labels: HashMap<String, String>,
    pub session_folder: Option<String>,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}
#[derive(Debug, Clone, Serialize)]
pub struct AddPathResult {
    pub chunks_added: usize,
    pub files_skipped: usize,
    pub points: usize,
}

#[derive(Debug, Clone, Default)]
pub struct SearchRequest {
    pub name: String,
    pub content: String,
    pub vector: Option<Vec<f32>>,
    pub top_k: usize,
    pub filters: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub point: Point,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum DeleteTarget {
    Store,
    Point(VectorID),
}
#[derive(Debug, Clone)]
pub struct DeleteRequest {
    pub name: String,
    pub target: DeleteTarget,
}
#[derive(Debug, Clone, Serialize)]
pub struct DeleteResult {
    pub id: Option<VectorID>,
    pub points: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoreSummary {
    pub name: String,
    pub metric: DistanceMetric,
    pub dimensions: usize,
    pub points: usize,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct EphemeralSearchRequest {
    pub corpus_path: PathBuf,
    /// Number of whitespace-delimited words per chunk.
    pub chunk_size: usize,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub query: String,
    pub top_k: usize,
}

#[allow(clippy::too_many_arguments)]
fn content_metadata(
    document_id: DocumentId,
    source_uri: SourceUri,
    chunk_index: usize,
    content: String,
    labels: HashMap<String, String>,
    path: Option<String>,
    start_line: Option<usize>,
    end_line: Option<usize>,
    language: Option<String>,
    session_folder: Option<String>,
) -> ChunkMetadata {
    ChunkMetadata {
        document_id,
        source_uri,
        chunk_index,
        content_hash: ContentHash(blake3::hash(content.as_bytes()).to_string()),
        content,
        labels,
        path,
        start_line,
        end_line,
        language,
        session_folder,
    }
}

fn search_store(
    store: &VectorStore,
    vector: Vec<f32>,
    top_k: usize,
    filters: &HashMap<String, String>,
) -> Result<Vec<SearchHit>, MainError> {
    let query = Point {
        id: VectorID::new(),
        vec: vector,
        metadata: ChunkMetadata::default(),
    };
    Ok(store
        .get_top_k_filtered_with_scores(&query, top_k, filters)?
        .into_iter()
        .map(|(point, score)| SearchHit { point, score })
        .collect())
}

fn validate_embeddings(
    expected_count: usize,
    dimensions: usize,
    embeddings: &[Vec<f32>],
) -> Result<(), MainError> {
    if embeddings.len() != expected_count {
        return Err(ApiError::EmbeddingCountMismatch {
            expected: expected_count,
            actual: embeddings.len(),
        }
        .into());
    }
    for (embedding_index, embedding) in embeddings.iter().enumerate() {
        if embedding.len() != dimensions {
            return Err(ApiError::EmbeddingDimMismatch {
                embedding_index,
                expected: dimensions,
                actual: embedding.len(),
            }
            .into());
        }
        if let Some(value_index) = embedding.iter().position(|value| !value.is_finite()) {
            return Err(ApiError::NonFiniteEmbedding {
                embedding_index,
                value_index,
            }
            .into());
        }
    }
    Ok(())
}

fn require_positive(value: usize, field: &'static str) -> Result<(), MainError> {
    if value == 0 {
        invalid(format!("{field} must be greater than zero"))
    } else {
        Ok(())
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, MainError> {
    Err(invalid_error(message))
}

fn invalid_error(message: impl Into<String>) -> MainError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn emit(observer: Option<&ProgressObserver>, event: ApplicationEvent) {
    if let Some(observer) = observer {
        observer(event);
    }
}
