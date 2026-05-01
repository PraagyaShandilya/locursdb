# locursdb

`locursdb` is a small Rust vector-store project for experimenting with local embedding ingestion and retrieval.

## What it does

The project currently:
- reads source text from `corpus/sample.txt`
- splits the text into fixed-size word chunks
- sends those chunks to the OpenRouter embeddings API
- stores the returned vectors in an in-memory `VectorStore`
- embeds a sample query
- runs a top-k nearest-neighbor lookup against the stored vectors
- decodes and prints the matching chunk text

## How it works

### Core pieces
- `src/store.rs` — in-memory vector store and top-k retrieval
- `src/distance.rs` — distance/similarity scoring
- `src/point.rs` — vector point and metadata types
- `src/api.rs` — OpenRouter embedding client and batch calls
- `src/ingest.rs` — text file loading and chunking
- `src/config.rs` — `.env`-backed runtime configuration
- `src/app.rs` — main ingestion/query workflow
- `src/main.rs` — thin executable entrypoint

### Ingestion flow
1. Load settings from `.env`
2. Read `corpus/sample.txt`
3. Split text into chunks using `CHUNK_SIZE`
4. Request embeddings in batches using `BATCH_SIZE`
5. Insert vectors plus metadata into `VectorStore`

### Retrieval flow
1. Embed a query string
2. Compare the query vector against stored vectors
3. Return the top `k` nearest points
4. Decode stored Base64 chunk content and print it

## Configuration

Create a `.env` file in the project root with:
- `OPENROUTER_API_KEY`
- `MODEL_NAME`
- `BATCH_SIZE`
- `CHUNK_SIZE`
- `DIMENSIONS`

## Running

```bash
cargo run
```

## Testing

```bash
cargo test
```

## Benchmarking

```bash
cargo bench --bench store_benches
```

## Current state

This is still an experimental, in-memory vector database prototype. It is useful for learning and iteration, not yet a production persistence or ANN indexing system.
