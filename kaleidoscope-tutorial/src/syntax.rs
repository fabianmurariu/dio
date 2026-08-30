use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "kaleidoscope.pest"]
pub(crate) struct KaleidoscopeParser;
