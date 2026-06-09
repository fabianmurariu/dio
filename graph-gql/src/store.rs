//! Process-global store of named raphtory graphs.

use std::collections::HashMap;
use std::sync::RwLock;

use raphtory::prelude::*;

/// A map of graph name -> raphtory [`Graph`], shared across requests.
#[derive(Default)]
pub struct GraphStore {
    graphs: RwLock<HashMap<String, Graph>>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a named graph.
    pub fn insert(&self, name: impl Into<String>, g: Graph) {
        self.graphs.write().unwrap().insert(name.into(), g);
    }

    /// Create an empty named graph.
    pub fn create(&self, name: impl Into<String>) {
        self.insert(name, Graph::new());
    }

    /// Run `f` with the named graph held under a read lock; `None` if absent.
    pub fn with<R>(&self, name: &str, f: impl FnOnce(&Graph) -> R) -> Option<R> {
        self.graphs.read().unwrap().get(name).map(f)
    }

    /// Add an edge to the named graph (raphtory mutates via interior mutability).
    pub fn add_edge(&self, name: &str, time: i64, src: u64, dst: u64) -> Result<(), String> {
        let graphs = self.graphs.read().unwrap();
        let g = graphs
            .get(name)
            .ok_or_else(|| format!("no graph named {name:?}"))?;
        g.add_edge(time, src, dst, NO_PROPS, None)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}
