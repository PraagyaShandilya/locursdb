use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use locursdb::{
    ChunkMetadata, ContentHash, DistanceMetric, DocumentId, Point, SourceUri, VectorID,
    VectorStore,
};

fn make_metadata(document_id: &str, source_uri: &str, chunk_index: usize) -> ChunkMetadata {
    ChunkMetadata {
        document_id: DocumentId(document_id.to_string()),
        source_uri: SourceUri(source_uri.to_string()),
        chunk_index,
        content_hash: ContentHash(format!("hash_{chunk_index}")),
    }
}

fn make_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| (((seed * 31 + i * 17) % 1_000) as f32) / 1_000.0)
        .collect()
}

fn make_point(dim: usize, seed: usize) -> Point {
    Point {
        id: VectorID::new(),
        vec: make_vec(dim, seed),
        metadata: make_metadata("doc_test", "memory://bench", seed),
    }
}

fn build_store(size: usize, dim: usize, metric: DistanceMetric) -> VectorStore {
    let mut store = VectorStore::new(metric);

    for i in 0..size {
        store
            .upsert(
                VectorID::new(),
                make_vec(dim, i),
                make_metadata("doc", "memory://bench", i),
            )
            .unwrap();
    }

    store
}

fn bench_get_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_top_k");

    for size in [100_usize, 1_000, 10_000] {
        for dim in [128_usize, 512] {
            let store = build_store(size, dim, DistanceMetric::Euclid);
            let query = make_point(dim, 999_999);

            group.bench_with_input(
                BenchmarkId::new(format!("euclid_size{size}_dim{dim}"), 5),
                &5_usize,
                |b, &k| {
                    b.iter(|| black_box(store.get_top_k(black_box(&query), black_box(k))));
                },
            );
        }
    }

    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");

    for size in [100_usize, 1_000, 10_000] {
        let mut store = VectorStore::new(DistanceMetric::Euclid);
        let target_id = VectorID::new();

        store
            .upsert(
                target_id,
                make_vec(512, 0),
                make_metadata("doc", "memory://bench", 0),
            )
            .unwrap();

        for i in 1..size {
            store
                .upsert(
                    VectorID::new(),
                    make_vec(512, i),
                    make_metadata("doc", "memory://bench", i),
                )
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::new("existing", size), &target_id, |b, id| {
            b.iter(|| black_box(store.get(black_box(id)).unwrap()));
        });
    }

    group.finish();
}

fn bench_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("upsert");

    for size in [100_usize, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("fresh_insert", size), &size, |b, &n| {
            b.iter(|| {
                let mut store = VectorStore::new(DistanceMetric::Euclid);
                for i in 0..n {
                    store
                        .upsert(
                            VectorID::new(),
                            make_vec(512, i),
                            make_metadata("doc", "memory://bench", i),
                        )
                        .unwrap();
                }
                black_box(store);
            });
        });
    }

    group.finish();
}

fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    for dim in [128_usize, 512, 1536] {
        let p1 = make_point(dim, 1);
        let p2 = make_point(dim, 2);

        group.bench_with_input(BenchmarkId::new("euclid", dim), &dim, |b, _| {
            b.iter(|| black_box(DistanceMetric::Euclid.distance(black_box(&p1), black_box(&p2))));
        });

        group.bench_with_input(BenchmarkId::new("cos", dim), &dim, |b, _| {
            b.iter(|| black_box(DistanceMetric::Cos.distance(black_box(&p1), black_box(&p2))));
        });

        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |b, _| {
            b.iter(|| black_box(DistanceMetric::Dot.distance(black_box(&p1), black_box(&p2))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_get_top_k,
    bench_get,
    bench_upsert,
    bench_distance_metrics
);
criterion_main!(benches);
