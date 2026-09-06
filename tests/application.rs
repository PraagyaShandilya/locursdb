use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use locursdb::{
    ApiError, DistanceMetric, MainError, PseudoEmbedder, SourceUri,
    application::{
        AddContentRequest, AddPathRequest, Application, ApplicationEvent, CreateStoreRequest,
        DeleteRequest, DeleteTarget, EmbedFuture, Embedder, EphemeralSearchRequest,
        ProgressObserver, SearchRequest,
    },
};
use ulid::Ulid;

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!("locursdb-{label}-{}", Ulid::new()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Default)]
struct ValidEmbedder;

impl Embedder for ValidEmbedder {
    fn embed<'a>(
        &'a self,
        inputs: Vec<String>,
        dimensions: usize,
        _observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a> {
        Box::pin(async move {
            Ok(inputs
                .into_iter()
                .map(|input| {
                    let mut vector = vec![0.0; dimensions];
                    if dimensions != 0 {
                        vector[0] = input.bytes().map(f32::from).sum::<f32>().max(1.0);
                    }
                    vector
                })
                .collect())
        })
    }
}

#[derive(Debug)]
struct NeverEmbedder;

impl Embedder for NeverEmbedder {
    fn embed<'a>(
        &'a self,
        _inputs: Vec<String>,
        _dimensions: usize,
        _observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a> {
        panic!("explicit vectors must bypass the embedder")
    }
}

#[derive(Debug)]
struct QueuedEmbedder(Mutex<VecDeque<Vec<Vec<f32>>>>);

impl QueuedEmbedder {
    fn new(responses: impl IntoIterator<Item = Vec<Vec<f32>>>) -> Self {
        Self(Mutex::new(responses.into_iter().collect()))
    }
}

impl Embedder for QueuedEmbedder {
    fn embed<'a>(
        &'a self,
        _inputs: Vec<String>,
        _dimensions: usize,
        _observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a> {
        let response = self.0.lock().unwrap().pop_front().unwrap();
        Box::pin(async move { Ok(response) })
    }
}

fn create(application: &Application, name: &str, dimensions: usize) {
    application
        .create(CreateStoreRequest {
            name: name.to_string(),
            metric: DistanceMetric::Euclid,
            dimensions,
        })
        .unwrap();
}

#[tokio::test]
async fn content_and_path_ingest_search_filters_and_preserves_metadata() {
    let root = TempDir::new("application-content-path");
    let corpus = TempDir::new("application-corpus");
    let source = corpus.0.join("notes.rs");
    fs::write(&source, "fn alpha() {}\nfn beta() {}\n").unwrap();
    let application = Application::at_root(root.0.clone(), Arc::new(ValidEmbedder));
    create(&application, "docs", 2);

    let added = application
        .add_content(
            AddContentRequest {
                name: "docs".to_string(),
                content: "manual entry".to_string(),
                vector: Some(vec![1.0, 0.0]),
                source_uri: Some(SourceUri("test://manual".to_string())),
                labels: HashMap::from([("kind".to_string(), "manual".to_string())]),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(added.points, 1);

    let added = application
        .add_path(
            AddPathRequest {
                name: "docs".to_string(),
                path: source.clone(),
                labels: HashMap::from([("kind".to_string(), "source".to_string())]),
                session_folder: Some("session-a".to_string()),
                chunk_size: 1_000,
                chunk_overlap: 0,
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(added.chunks_added, 1);
    assert_eq!(added.points, 2);

    let hits = application
        .search(
            SearchRequest {
                name: "docs".to_string(),
                vector: Some(vec![0.0, 0.0]),
                top_k: 5,
                filters: HashMap::from([("kind".to_string(), "source".to_string())]),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let metadata = &hits[0].point.metadata;
    assert_eq!(metadata.source_uri.0, source.to_string_lossy());
    assert_eq!(
        metadata.path.as_deref(),
        Some(source.to_string_lossy().as_ref())
    );
    assert_eq!(metadata.language.as_deref(), Some("rust"));
    assert_eq!(metadata.start_line, Some(1));
    assert_eq!(metadata.session_folder.as_deref(), Some("session-a"));
    assert!(!metadata.content_hash.0.is_empty());
}

#[tokio::test]
async fn explicit_vectors_bypass_embedding_and_are_validated_by_the_service() {
    let root = TempDir::new("application-explicit");
    let application = Application::at_root(root.0.clone(), Arc::new(NeverEmbedder));
    create(&application, "docs", 2);

    application
        .add_content(
            AddContentRequest {
                name: "docs".to_string(),
                content: "".to_string(),
                vector: Some(vec![1.0, 2.0]),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    let hits = application
        .search(
            SearchRequest {
                name: "docs".to_string(),
                vector: Some(vec![1.0, 2.0]),
                top_k: 1,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].point.metadata.content, "");
    assert_eq!(hits[0].point.metadata.source_uri.0, "content://add");

    let no_hits = application
        .search(
            SearchRequest {
                name: "docs".to_string(),
                vector: Some(vec![1.0, 2.0]),
                top_k: 0,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    assert!(no_hits.is_empty());

    for vector in [vec![1.0], vec![1.0, f32::INFINITY]] {
        let error = application
            .add_content(
                AddContentRequest {
                    name: "docs".to_string(),
                    vector: Some(vector),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                MainError::ApiError(ApiError::EmbeddingDimMismatch { .. })
            ) || matches!(
                error,
                MainError::ApiError(ApiError::NonFiniteEmbedding { .. })
            )
        );
    }
    assert_eq!(application.list().unwrap()[0].points, 1);
}

#[tokio::test]
async fn embedded_content_is_retrieved_by_an_embedded_query() {
    let root = TempDir::new("application-embedded-search");
    let embedder = QueuedEmbedder::new([vec![vec![1.0, 0.0]], vec![vec![1.0, 0.0]]]);
    let application = Application::at_root(root.0.clone(), Arc::new(embedder));
    create(&application, "docs", 2);

    application
        .add_content(
            AddContentRequest {
                name: "docs".to_string(),
                content: "embedded document".to_string(),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    let hits = application
        .search(
            SearchRequest {
                name: "docs".to_string(),
                content: "embedded query".to_string(),
                top_k: 1,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].point.metadata.content, "embedded document");
    assert_eq!(hits[0].point.vec, vec![1.0, 0.0]);
}

#[tokio::test]
async fn point_delete_is_reflected_in_list() {
    let root = TempDir::new("application-delete-point");
    let application = Application::at_root(root.0.clone(), Arc::new(NeverEmbedder));
    create(&application, "docs", 2);
    let added = application
        .add_content(
            AddContentRequest {
                name: "docs".to_string(),
                vector: Some(vec![1.0, 0.0]),
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

    let deleted = application
        .delete(DeleteRequest {
            name: "docs".to_string(),
            target: DeleteTarget::Point(added.id),
        })
        .unwrap();

    assert_eq!(deleted.id, Some(added.id));
    assert_eq!(deleted.points, Some(0));
    assert_eq!(application.list().unwrap()[0].points, 0);
}

#[tokio::test]
async fn malformed_provider_output_is_rejected_before_persistence() {
    enum Expected {
        Count,
        Dimensions,
        NonFinite,
    }

    for (response, expected) in [
        (Vec::<Vec<f32>>::new(), Expected::Count),
        (vec![vec![1.0]], Expected::Dimensions),
        (vec![vec![1.0, f32::NAN]], Expected::NonFinite),
    ] {
        let root = TempDir::new("application-malformed");
        let application =
            Application::at_root(root.0.clone(), Arc::new(QueuedEmbedder::new([response])));
        create(&application, "docs", 2);
        let error = application
            .add_content(
                AddContentRequest {
                    name: "docs".to_string(),
                    content: "embed me".to_string(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(match expected {
            Expected::Count => matches!(
                error,
                MainError::ApiError(ApiError::EmbeddingCountMismatch { .. })
            ),
            Expected::Dimensions => matches!(
                error,
                MainError::ApiError(ApiError::EmbeddingDimMismatch { .. })
            ),
            Expected::NonFinite => matches!(
                error,
                MainError::ApiError(ApiError::NonFiniteEmbedding { .. })
            ),
        });
        assert_eq!(application.list().unwrap()[0].points, 0);
    }
}

#[tokio::test]
async fn failure_on_a_later_file_does_not_save_earlier_file_points() {
    let root = TempDir::new("application-atomic");
    let corpus = TempDir::new("application-atomic-corpus");
    fs::write(corpus.0.join("a.txt"), "alpha").unwrap();
    fs::write(corpus.0.join("b.txt"), "beta").unwrap();
    let embedder = QueuedEmbedder::new([vec![vec![1.0, 0.0]], vec![]]);
    let application = Application::at_root(root.0.clone(), Arc::new(embedder));
    create(&application, "docs", 2);

    let result = application
        .add_path(
            AddPathRequest {
                name: "docs".to_string(),
                path: corpus.0.clone(),
                labels: HashMap::new(),
                session_folder: None,
                chunk_size: 100,
                chunk_overlap: 0,
            },
            None,
        )
        .await;
    assert!(matches!(
        result,
        Err(MainError::ApiError(ApiError::EmbeddingCountMismatch { .. }))
    ));
    assert_eq!(application.list().unwrap()[0].points, 0);
}

#[tokio::test]
async fn typed_requests_are_validated_before_work() {
    let root = TempDir::new("application-validation");
    let application = Application::at_root(root.0.clone(), Arc::new(NeverEmbedder));

    assert!(
        application
            .create(CreateStoreRequest {
                name: "bad/name".to_string(),
                metric: DistanceMetric::Euclid,
                dimensions: 2,
            })
            .is_err()
    );
    assert!(
        application
            .create(CreateStoreRequest {
                name: "docs".to_string(),
                metric: DistanceMetric::Euclid,
                dimensions: 0,
            })
            .is_err()
    );
    assert!(!root.0.join("docs").exists());

    create(&application, "docs", 2);
    let error = application
        .add_path(
            AddPathRequest {
                name: "docs".to_string(),
                path: root.0.join("unused.txt"),
                labels: HashMap::new(),
                session_folder: None,
                chunk_size: 10,
                chunk_overlap: 10,
            },
            None,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("overlap"));
}

#[tokio::test]
async fn ephemeral_workflow_is_in_memory_preserves_word_chunks_and_reports_progress() {
    let root = TempDir::new("application-ephemeral-root");
    let corpus = TempDir::new("application-ephemeral-corpus");
    let source = corpus.0.join("words.txt");
    fs::write(&source, "one two three four five").unwrap();
    let application = Application::at_root(root.0.join("stores"), Arc::new(PseudoEmbedder));
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed_events = events.clone();
    let observer = move |event| observed_events.lock().unwrap().push(event);

    let hits = application
        .ingest_and_search(
            EphemeralSearchRequest {
                corpus_path: source.clone(),
                chunk_size: 2,
                dimensions: 4,
                metric: DistanceMetric::Euclid,
                query: "one".to_string(),
                top_k: 3,
            },
            Some(&observer),
        )
        .await
        .unwrap();

    let mut contents = hits
        .iter()
        .map(|hit| hit.point.metadata.content.clone())
        .collect::<Vec<_>>();
    contents.sort();
    assert_eq!(contents, ["five", "one two", "three four"]);
    assert!(hits.iter().all(|hit| {
        hit.point.metadata.source_uri.0 == source.to_string_lossy()
            && hit.point.metadata.path.as_deref() == Some(source.to_string_lossy().as_ref())
    }));
    assert!(!root.0.join("stores").exists());

    let events = events.lock().unwrap();
    assert!(events.contains(&ApplicationEvent::EmbeddingProgress {
        completed_batches: 0,
        total_batches: 1,
    }));
    assert!(events.contains(&ApplicationEvent::EmbeddingProgress {
        completed_batches: 1,
        total_batches: 1,
    }));
}
