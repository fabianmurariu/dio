//! A minimal [`ContextProvider`] so datafusion's `SqlToRel` can resolve table
//! names to schemas and the aggregate functions we support (count/min/max/sum).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result, TableReference};
use datafusion_expr::logical_plan::builder::LogicalTableSource;
use datafusion_expr::planner::ContextProvider;
use datafusion_expr::{AggregateUDF, HigherOrderUDF, ScalarUDF, TableSource, WindowUDF};
use datafusion_functions::string::octet_length;
use datafusion_functions_aggregate::average::avg_udaf;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_functions_aggregate::min_max::{max_udaf, min_udaf};
use datafusion_functions_aggregate::sum::sum_udaf;

/// Maps table names to arrow schemas + the supported scalar / aggregate UDFs,
/// for logical planning.
pub struct Catalog {
    tables: HashMap<String, SchemaRef>,
    functions: HashMap<String, Arc<ScalarUDF>>,
    aggregates: HashMap<String, Arc<AggregateUDF>>,
    options: ConfigOptions,
}

impl Default for Catalog {
    fn default() -> Self {
        let functions = [octet_length()]
            .into_iter()
            .map(|udf| (udf.name().to_string(), udf))
            .collect();
        let aggregates = [count_udaf(), sum_udaf(), min_udaf(), max_udaf(), avg_udaf()]
            .into_iter()
            .map(|udaf| (udaf.name().to_string(), udaf))
            .collect();
        Self {
            tables: HashMap::new(),
            functions,
            aggregates,
            options: ConfigOptions::default(),
        }
    }
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

    fn get_function_meta(&self, name: &str) -> Option<Arc<ScalarUDF>> {
        self.functions.get(name).cloned()
    }

    fn get_aggregate_meta(&self, name: &str) -> Option<Arc<AggregateUDF>> {
        self.aggregates.get(name).cloned()
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
        self.functions.keys().cloned().collect()
    }

    fn udaf_names(&self) -> Vec<String> {
        self.aggregates.keys().cloned().collect()
    }

    fn udwf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        Vec::new()
    }
}
