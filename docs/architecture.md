# Architecture

The project is migrating UI adapters onto a shared, typed in-process application API. The first
phase moves the CLI; the TUI remains on the legacy workflow until its migration branch lands.
Dependencies point inward from adapters to workflows and from workflows to implementation modules.

```mermaid
graph LR
    CLI[CLI: parse / present / compose] --> Application[application workflows]
    Application --> Repository[store_repository]
    Application --> Ingest[ingest]
    Application --> Embedder[Embedder]
    Application --> Vector[vector]
    Embedder --> Local[embedding::local]
    Embedder --> Remote[embedding::ApiClient]
    Repository --> Vector
    TUI[TUI: legacy, migration next] -.-> Legacy[legacy ingest / embedding APIs]
```

## Ownership and dependency rules

- `cli` owns argument parsing, provider composition, and JSON presentation. It does not open stores,
  inspect persistence configuration, validate vector dimensions, ingest files, or perform searches.
- `application` is the typed workflow boundary. It validates requests and embedding output,
  discovers store dimensions, constructs metadata, coordinates ingest/search, and saves only after a
  complete path-ingest batch succeeds. This prevents partial saves on embedding failures, not
  crash-atomic disk writes or concurrent-writer conflicts.
- `store_repository` owns store-root resolution, store-name validation, persistence paths, JSON
  persistence, and store lifecycle operations.
- `ingest::source` owns source discovery, ignore handling, supported languages, file reading,
  character chunking, overlap, and line ranges. Legacy `Ingest` retains word chunking for the
  ephemeral workflow.
- `Embedder` is an in-process provider abstraction. It accepts the dimensions required by each
  workflow and can report provider-neutral completed/total batch events. Existing provider progress
  channel APIs remain supported. Concrete adapters (`OpenRouterEmbedder`, `PseudoEmbedder`) live in
  `embedding::provider`, not in the workflow implementation.
- `ApplicationEvent` observers are synchronous, optional callbacks. Stages announce work starting;
  the workflow's `Result` indicates success or failure. Frontends own event formatting and channels.
- The CLI's OpenRouter adapter loads configuration lazily on the first actual embedding request.
  Consequently create/list/delete, pseudo embeddings, and explicit-vector add/search do not require
  provider configuration. `ApiClient::with_dimensions` reuses its reqwest client while honoring each
  store's dimensions.
- `vector` owns vector-store data structures, vector validation, and search behavior.

`Application::ingest_and_search` creates its store and metadata entirely in memory. It uses the
actual corpus path as source metadata and never creates repository files.

## Migration status

The CLI now calls `Application` exclusively for workflows. `app.rs` and `tui.rs` intentionally remain
unchanged and continue using the legacy progress and ingest APIs. Their next migration should compose
an `Embedder`, translate `ApplicationEvent` values into TUI messages, and call
`Application::ingest_and_search` rather than duplicating orchestration.
