# locursdb

`locursdb` is a small Rust vector-store project for experimenting with local embedding ingestion and retrieval.

## Configuration

Create a `.env` file in the project root with:
- `OPENROUTER_API_KEY`
- `MODEL_NAME`
- `BATCH_SIZE`
- `EMBEDDING_CONCURRENCY`
- `CHUNK_SIZE`
- `DIMENSIONS`

Optional:
- `CORPUS_PATH` defaults to `corpus/sample.txt`
- `TOP_K` defaults to `5`

## Running

```bash
cargo run
```

During embedding and query processing, the TUI stays open and displays embedding batch progress. Detailed progress traces are also written to:

```text
log/embedding.log
```

The file is overwritten on each run.

### TUI controls

- `Tab`: switch input field
- `Enter`: run search / close results
- `Esc` / `q`: quit

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
