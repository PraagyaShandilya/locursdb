use std::{env, fs, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{DistanceMetric, MainError, VectorStore};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct StoreConfig {
    pub(crate) name: String,
    pub(crate) metric: DistanceMetric,
    pub(crate) dimensions: usize,
}

#[derive(Debug)]
pub(crate) struct CreatedStore {
    pub(crate) config: StoreConfig,
    pub(crate) path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct OpenStore {
    pub(crate) config: StoreConfig,
    pub(crate) store: VectorStore,
    path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct StoreSummary {
    pub(crate) name: String,
    pub(crate) metric: DistanceMetric,
    pub(crate) dimensions: usize,
    pub(crate) points: usize,
    pub(crate) path: PathBuf,
}

#[derive(Debug)]
pub struct StoreRepository {
    root: PathBuf,
}

impl StoreRepository {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn from_environment() -> Self {
        Self::new(
            env::var("LOCURSDB_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(".locursdb")),
        )
    }

    pub(crate) fn create(
        &self,
        name: String,
        metric: DistanceMetric,
        dimensions: usize,
    ) -> Result<CreatedStore, MainError> {
        validate_store_name(&name)?;
        let path = self.store_dir(&name);
        fs::create_dir_all(&path)?;
        let config = StoreConfig {
            name,
            metric,
            dimensions,
        };
        self.save_config(&path, &config)?;
        if !self.points_path(&path).exists() {
            self.save_at(
                &path,
                &VectorStore::with_dimensions(config.metric, dimensions),
            )?;
        }
        Ok(CreatedStore { config, path })
    }

    pub(crate) fn open(&self, name: &str) -> Result<OpenStore, MainError> {
        validate_store_name(name)?;
        let path = self.store_dir(name);
        let config = self.load_config(&path)?;
        let store = self.load_at(&path, &config)?;
        Ok(OpenStore {
            config,
            store,
            path,
        })
    }

    pub(crate) fn save(&self, open: &OpenStore) -> Result<(), MainError> {
        self.save_at(&open.path, &open.store)
    }

    pub(crate) fn delete(&self, name: &str) -> Result<(), MainError> {
        validate_store_name(name)?;
        fs::remove_dir_all(self.store_dir(name))?;
        Ok(())
    }

    pub(crate) fn list(&self) -> Result<Vec<StoreSummary>, MainError> {
        let mut stores = Vec::new();
        if self.root.exists() {
            for entry in fs::read_dir(&self.root)? {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                let path = entry.path();
                if let Ok(config) = self.load_config(&path) {
                    let points = self
                        .load_at(&path, &config)
                        .map(|store| store.len())
                        .unwrap_or(0);
                    stores.push(StoreSummary {
                        name: config.name,
                        metric: config.metric,
                        dimensions: config.dimensions,
                        points,
                        path,
                    });
                }
            }
        }
        Ok(stores)
    }

    fn store_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn config_path(&self, path: &std::path::Path) -> PathBuf {
        path.join("config.json")
    }

    fn points_path(&self, path: &std::path::Path) -> PathBuf {
        path.join("points.json")
    }

    fn save_config(&self, path: &std::path::Path, config: &StoreConfig) -> Result<(), MainError> {
        fs::write(self.config_path(path), serde_json::to_vec_pretty(config)?)?;
        Ok(())
    }

    fn load_config(&self, path: &std::path::Path) -> Result<StoreConfig, MainError> {
        Ok(serde_json::from_slice(&fs::read(self.config_path(path))?)?)
    }

    fn save_at(&self, path: &std::path::Path, store: &VectorStore) -> Result<(), MainError> {
        fs::write(self.points_path(path), serde_json::to_vec_pretty(store)?)?;
        Ok(())
    }

    fn load_at(
        &self,
        path: &std::path::Path,
        config: &StoreConfig,
    ) -> Result<VectorStore, MainError> {
        let points_path = self.points_path(path);
        if points_path.exists() {
            Ok(serde_json::from_slice(&fs::read(points_path)?)?)
        } else {
            Ok(VectorStore::with_dimensions(
                config.metric,
                config.dimensions,
            ))
        }
    }
}

fn validate_store_name(name: &str) -> Result<(), MainError> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name == "." || name == ".." {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "store name must be a simple path segment",
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use ulid::Ulid;

    use super::StoreRepository;
    use crate::{ChunkMetadata, DistanceMetric, VectorID};

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("locursdb-store-repository-{}", Ulid::new())))
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn repository_round_trip_preserves_points() {
        let root = TempRoot::new();
        let repository = StoreRepository::new(root.0.clone());
        let created = repository
            .create("test".to_string(), DistanceMetric::Euclid, 2)
            .unwrap();
        assert_eq!(created.path, root.0.join("test"));

        let mut open = repository.open("test").unwrap();
        open.store
            .upsert(VectorID::new(), vec![1.0, 2.0], ChunkMetadata::default())
            .unwrap();
        repository.save(&open).unwrap();

        repository
            .create("test".to_string(), DistanceMetric::Euclid, 2)
            .unwrap();
        assert_eq!(repository.open("test").unwrap().store.len(), 1);

        let stores = repository.list().unwrap();
        assert_eq!(stores.len(), 1);
        assert_eq!(stores[0].name, "test");
        assert_eq!(stores[0].points, 1);

        repository.delete("test").unwrap();
        assert!(!root.0.join("test").exists());
    }

    #[test]
    fn repository_rejects_non_segment_store_names() {
        let root = TempRoot::new();
        let repository = StoreRepository::new(root.0.clone());

        let create_error = repository
            .create("../escape".to_string(), DistanceMetric::Euclid, 2)
            .unwrap_err();
        let open_error = repository.open("../escape").unwrap_err();
        let delete_error = repository.delete("../escape").unwrap_err();

        for error in [create_error, open_error, delete_error] {
            assert!(error.to_string().contains("simple path segment"));
        }
        assert!(!root.0.exists());
    }
}
