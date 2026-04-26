use locursdb::{
    ChunkMetadata, ContentHash, DistanceMetric, DocumentId, Point, SourceUri, VectorID,
    VectorIDError, VectorStore,
};

fn make_metadata(document_id: &str, source_uri: &str, chunk_index: usize) -> ChunkMetadata {
    ChunkMetadata {
        document_id: DocumentId(document_id.to_string()),
        source_uri: SourceUri(source_uri.to_string()),
        chunk_index,
        content_hash: ContentHash(format!("hash_{chunk_index}")),
    }
}

fn make_point(vec: Vec<f32>) -> Point {
    Point {
        id: VectorID::new(),
        vec,
        metadata: make_metadata("doc_test", "memory://test", 0),
    }
}

fn assert_close(actual: f32, expected: f32) {
    let diff = (actual - expected).abs();
    assert!(
        diff < 1e-5,
        "expected {expected}, got {actual} (diff {diff})"
    );
}

#[test]
fn insert_and_count() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    store.upsert(
        VectorID::new(),
        vec![0.1_f32, 0.2, 0.3],
        make_metadata("doc_1", "memory://hello-world", 0),
    ).unwrap();
    assert_eq!(store.len(), 1);
}

#[test]
fn dimension_mismatch_rejected() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    store
        .upsert(
            VectorID::new(),
            vec![0.1, 0.2],
            make_metadata("doc_a", "memory://a", 0),
        )
        .unwrap();

    let err = store.upsert(
        VectorID::new(),
        vec![0.1, 0.2, 0.3],  // wrong dimension!
        make_metadata("doc_b", "memory://b", 0),
    ).unwrap_err();

    assert!(matches!(err, VectorIDError::DimMismatch { .. }));
}

#[test]
fn get_returns_inserted_point() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    let id = VectorID::new();
    let metadata = make_metadata("doc_42", "memory://vectors", 3);

    store
        .upsert(id.clone(), vec![1.0_f32, 2.0, 3.0], metadata.clone())
        .unwrap();

    let point = store.get(&id).unwrap();

    assert_eq!(point.id, id);
    assert_eq!(point.vec, vec![1.0_f32, 2.0, 3.0]);
    assert_eq!(point.metadata, metadata);
}

#[test]
fn get_missing_id_returns_not_found() {
    let store = VectorStore::new(DistanceMetric::Euclid);
    let missing_id = VectorID::new();

    let err = store.get(&missing_id).unwrap_err();

    assert!(matches!(err, VectorIDError::NotFound(id) if id == missing_id.to_string()));
}

#[test]
fn delete_removes_only_matching_point() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    let keep_id = VectorID::new();
    let delete_id = VectorID::new();

    store
        .upsert(
            keep_id.clone(),
            vec![0.1_f32, 0.2],
            make_metadata("doc_keep", "memory://keep", 0),
        )
        .unwrap();
    store
        .upsert(
            delete_id.clone(),
            vec![0.3_f32, 0.4],
            make_metadata("doc_delete", "memory://delete", 1),
        )
        .unwrap();

    store.delete(delete_id.clone());

    assert_eq!(store.len(), 1);
    assert!(matches!(
        store.get(&delete_id),
        Err(VectorIDError::NotFound(id)) if id == delete_id.to_string()
    ));

    let remaining = store.get(&keep_id).unwrap();
    assert_eq!(remaining.id, keep_id);
    assert_eq!(remaining.vec, vec![0.1_f32, 0.2]);
}

#[test]
fn euclid_distance_matches_expected_value() {
    let point1 = make_point(vec![0.0, 0.0]);
    let point2 = make_point(vec![4.0, 5.0]);

    let distance = DistanceMetric::Euclid.distance(&point1, &point2);

    assert_close(distance, 41.0);
}

#[test]
fn cosine_distance_is_zero_for_identical_vectors() {
    let point1 = make_point(vec![1.0, 2.0, 3.0]);
    let point2 = make_point(vec![1.0, 2.0, 3.0]);

    let distance = DistanceMetric::Cos.distance(&point1, &point2);

    assert_close(distance, 0.0);
}

#[test]
fn cosine_distance_is_one_for_orthogonal_vectors() {
    let point1 = make_point(vec![1.0, 0.0]);
    let point2 = make_point(vec![0.0, 1.0]);

    let distance = DistanceMetric::Cos.distance(&point1, &point2);

    assert_close(distance, 1.0);
}

#[test]
fn get_top_k_returns_nearest_vectors_for_euclidean_distance() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);

    store
        .upsert(
            VectorID::new(),
            vec![0.0_f32, 0.0],
            make_metadata("doc_a", "memory://a", 0),
        )
        .unwrap();
    store
        .upsert(
            VectorID::new(),
            vec![1.0_f32, 1.0],
            make_metadata("doc_b", "memory://b", 1),
        )
        .unwrap();
    store
        .upsert(
            VectorID::new(),
            vec![3.0_f32, 3.0],
            make_metadata("doc_c", "memory://c", 2),
        )
        .unwrap();

    let query = make_point(vec![0.0_f32, 0.0]);
    let top_k = store.get_top_k(&query, 2);

    assert_eq!(top_k.len(), 2);
    assert_eq!(top_k[0], vec![0.0_f32, 0.0]);
    assert_eq!(top_k[1], vec![1.0_f32, 1.0]);
}
