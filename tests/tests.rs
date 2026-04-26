use locursdb::{DistanceMetric, Point, VectorID, VectorIDError, VectorStore};

fn make_point(id: &str, vec: Vec<f32>) -> Point {
    Point {
        id: VectorID::new(id),
        vec,
        metadata: serde_json::Value::Null,
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
        VectorID::new("doc_1"),
        vec![0.1_f32, 0.2, 0.3],
        serde_json::json!({"text": "hello world"}),
    ).unwrap();
    assert_eq!(store.len(), 1);
}

#[test]
fn dimension_mismatch_rejected() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    store.upsert(VectorID::new("a"), vec![0.1, 0.2], serde_json::Value::Null).unwrap();

    let err = store.upsert(
        VectorID::new("b"),
        vec![0.1, 0.2, 0.3],  // wrong dimension!
        serde_json::Value::Null,
    ).unwrap_err();

    assert!(matches!(err, VectorIDError::DimMismatch { .. }));
}

#[test]
fn get_returns_inserted_point() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    let id = VectorID::new("doc_42");
    let metadata = serde_json::json!({"topic": "vectors"});

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
    let missing_id = VectorID::new("missing");

    let err = store.get(&missing_id).unwrap_err();

    assert!(matches!(err, VectorIDError::NotFound(id) if id == "missing"));
}

#[test]
fn delete_removes_only_matching_point() {
    let mut store = VectorStore::new(DistanceMetric::Euclid);
    let keep_id = VectorID::new("keep");
    let delete_id = VectorID::new("delete");

    store
        .upsert(keep_id.clone(), vec![0.1_f32, 0.2], serde_json::json!({"doc": 1}))
        .unwrap();
    store
        .upsert(delete_id.clone(), vec![0.3_f32, 0.4], serde_json::json!({"doc": 2}))
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
    let point1 = make_point("a", vec![0.0, 0.0]);
    let point2 = make_point("b", vec![4.0, 5.0]);

    let distance = DistanceMetric::Euclid.distance(&point1, &point2);

    assert_close(distance, 41.0);
}

#[test]
fn cosine_distance_is_zero_for_identical_vectors() {
    let point1 = make_point("a", vec![1.0, 2.0, 3.0]);
    let point2 = make_point("b", vec![1.0, 2.0, 3.0]);

    let distance = DistanceMetric::Cos.distance(&point1, &point2);

    assert_close(distance, 0.0);
}

#[test]
fn cosine_distance_is_one_for_orthogonal_vectors() {
    let point1 = make_point("a", vec![1.0, 0.0]);
    let point2 = make_point("b", vec![0.0, 1.0]);

    let distance = DistanceMetric::Cos.distance(&point1, &point2);

    assert_close(distance, 1.0);
}

