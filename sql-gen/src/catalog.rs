//! A minimal [`ContextProvider`] so datafusion's `SqlToRel` can resolve table
//! names to schemas. It knows nothing else — no functions, no variables — which
//! is all our "very simple queries" need.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result, TableReference};
use datafusion_expr::logical_plan::builder::LogicalTableSource;
use datafusion_expr::planner::ContextProvider;
use datafusion_expr::{AggregateUDF, HigherOrderUDF, ScalarUDF, TableSource, WindowUDF};

/// Maps table names to arrow schemas for logical planning.
#[derive(Default)]
pub struct Catalog {
    tables: HashMap<String, SchemaRef>,
    options: ConfigOptions,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_table(mut self, name: impl Into<String>, schema: SchemaRef) -> Self {
        self.tables.insert(name.into(), schema);
        self
    }
}

impl ContextProvider for Catalog {
    fn get_table_source(&self, name: TableReference) -> Result<Arc<dyn TableSource>> {
        match self.tables.get(name.table()) {
            Some(schema) => Ok(Arc::new(LogicalTableSource::new(schema.clone()))),
            None => Err(DataFusionError::Plan(format!(
                "table not found: {}",
                name.table()
            ))),
        }
    }

    fn get_function_meta(&self, _name: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
        None
    }

    fn get_window_meta(&self, _name: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_higher_order_meta(&self, _name: &str) -> Option<Arc<HigherOrderUDF>> {
        None
    }

    fn get_variable_type(&self, _variable_names: &[String]) -> Option<DataType> {
        None
    }

    fn options(&self) -> &ConfigOptions {
        &self.options
    }

    fn udf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn udaf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn udwf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        Vec::new()
    }
}
