use std::{fs, path::PathBuf, process::Command};

use serde_json::Value;
use ulid::Ulid;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("locursdb-cli-test-{}", Ulid::new())))
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(root: &TempRoot, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_locursdb"))
        .env("LOCURSDB_ROOT", &root.0)
        .args(args)
        .output()
        .unwrap()
}

fn success_json(root: &TempRoot, args: &[&str]) -> Value {
    let output = run(root, args);
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn pseudo_embedding_store_round_trip() {
    let root = TempRoot::new();

    let created = success_json(&root, &["create", "--name", "test", "--dimensions", "8"]);
    assert_eq!(created["command"], "create");

    let added = success_json(
        &root,
        &[
            "add",
            "--name",
            "test",
            "--content",
            "Rust ownership prevents data races",
            "--embed",
            "pseudo",
            "--labels",
            "language:rust",
        ],
    );
    assert_eq!(added["points"], 1);

    let searched = success_json(
        &root,
        &[
            "search",
            "--name",
            "test",
            "--content",
            "safe Rust",
            "--embed",
            "pseudo",
            "--filter",
            "language:rust",
            "--top-k",
            "1",
        ],
    );
    assert_eq!(
        searched["results"][0]["content"],
        "Rust ownership prevents data races"
    );

    let listed = success_json(&root, &["list"]);
    assert_eq!(listed["stores"][0]["name"], "test");
    assert_eq!(listed["stores"][0]["points"], 1);

    let deleted = success_json(&root, &["delete", "--name", "test", "--store"]);
    assert_eq!(deleted["deleted"], "store");
    assert!(!root.0.join("test").exists());
}

#[test]
fn explicit_vectors_bypass_invalid_embed_and_use_default_source() {
    let root = TempRoot::new();
    success_json(&root, &["create", "--name", "vectors", "--dimensions", "2"]);

    let added = success_json(
        &root,
        &[
            "add",
            "--name",
            "vectors",
            "--content",
            "explicit",
            "--vector",
            "1,2",
            "--embed",
            "not-a-provider",
        ],
    );
    assert_eq!(added["points"], 1);

    let searched = success_json(
        &root,
        &[
            "search",
            "--name",
            "vectors",
            "--vector",
            "1,2",
            "--embed",
            "not-a-provider",
            "--top-k",
            "1",
        ],
    );
    assert_eq!(searched["results"][0]["content"], "explicit");
    assert_eq!(
        searched["results"][0]["metadata"]["source_uri"],
        "cli://add"
    );
}

#[test]
fn path_with_vector_still_pseudo_embeds_every_chunk() {
    let root = TempRoot::new();
    fs::create_dir_all(&root.0).unwrap();
    let source = root.0.join("source.rs");
    fs::write(&source, "alpha\nbeta\n").unwrap();
    success_json(&root, &["create", "--name", "code", "--dimensions", "4"]);

    let added = success_json(
        &root,
        &[
            "add",
            "--name",
            "code",
            "--path",
            source.to_str().unwrap(),
            "--vector",
            "1",
            "--embed",
            "pseudo",
            "--chunk-size",
            "6",
            "--chunk-overlap",
            "0",
        ],
    );
    assert_eq!(added["chunks_added"], 2);
    assert_eq!(added["points"], 2);
}

#[test]
fn errors_are_json_on_stderr_with_nonzero_status() {
    let root = TempRoot::new();
    let output = run(&root, &["unknown"]);

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["ok"], false);
    assert!(error["error"].as_str().unwrap().contains("unknown command"));
}
