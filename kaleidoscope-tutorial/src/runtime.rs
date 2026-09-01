//! Chapter 4's interactive session and typed host environment.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;

use rust_lms::ffi::ExternRef;
use rust_lms::prelude::{Compiler, Ctx, Var, call_extern1, extern_fn};

use crate::ast::{Item, Program};
use crate::codegen::{CodegenError, compile_top_level, validate};
use crate::lexer::{Position, Span};
use crate::parser::{ParseError, ParserState};

#[extern_fn]
extern "C" fn host_sin(value: f64) -> f64 {
    value.sin()
}

#[extern_fn]
extern "C" fn host_cos(value: f64) -> f64 {
    value.cos()
}

#[extern_fn]
extern "C" fn host_exp(value: f64) -> f64 {
    value.exp()
}

#[extern_fn]
extern "C" fn host_log(value: f64) -> f64 {
    value.ln()
}

#[extern_fn]
extern "C" fn host_sqrt(value: f64) -> f64 {
    value.sqrt()
}

#[extern_fn]
extern "C" fn host_putchard(value: f64) -> f64 {
    let character = char::from(value as u8);
    let _ = write!(std::io::stderr().lock(), "{character}");
    0.0
}

/// The host functions made available to an `extern` declaration in Chapter 4.
pub const STANDARD_EXTERN_NAMES: &[&str] = &["sin", "cos", "exp", "log", "sqrt", "putchard"];

/// A typed rust-lms handle for one standard unary host function.
#[derive(Clone, Copy)]
pub(crate) enum HostFunctionRef {
    Sin(ExternRef<HostSinExtern>),
    Cos(ExternRef<HostCosExtern>),
    Exp(ExternRef<HostExpExtern>),
    Log(ExternRef<HostLogExtern>),
    Sqrt(ExternRef<HostSqrtExtern>),
    Putchard(ExternRef<HostPutchardExtern>),
}

impl HostFunctionRef {
    pub(crate) fn emit_call(self, ctx: &mut Ctx, argument: Var<f64>) -> Var<f64> {
        match self {
            Self::Sin(function) => ctx.bind(call_extern1(function, argument)),
            Self::Cos(function) => ctx.bind(call_extern1(function, argument)),
            Self::Exp(function) => ctx.bind(call_extern1(function, argument)),
            Self::Log(function) => ctx.bind(call_extern1(function, argument)),
            Self::Sqrt(function) => ctx.bind(call_extern1(function, argument)),
            Self::Putchard(function) => ctx.bind(call_extern1(function, argument)),
        }
    }
}

pub(crate) fn standard_extern_arity(name: &str) -> Option<usize> {
    STANDARD_EXTERN_NAMES.contains(&name).then_some(1)
}

pub(crate) fn register_standard_externs(
    compiler: &mut Compiler,
) -> HashMap<String, HostFunctionRef> {
    HashMap::from([
        (
            "sin".into(),
            HostFunctionRef::Sin(compiler.extern_fn::<HostSinExtern>()),
        ),
        (
            "cos".into(),
            HostFunctionRef::Cos(compiler.extern_fn::<HostCosExtern>()),
        ),
        (
            "exp".into(),
            HostFunctionRef::Exp(compiler.extern_fn::<HostExpExtern>()),
        ),
        (
            "log".into(),
            HostFunctionRef::Log(compiler.extern_fn::<HostLogExtern>()),
        ),
        (
            "sqrt".into(),
            HostFunctionRef::Sqrt(compiler.extern_fn::<HostSqrtExtern>()),
        ),
        (
            "putchard".into(),
            HostFunctionRef::Putchard(compiler.extern_fn::<HostPutchardExtern>()),
        ),
    ])
}

/// A Chapter 4 REPL state.
///
/// Definitions and prototypes persist between submissions. Each top-level
/// expression is compiled into a fresh, owner-checked native module containing
/// the accumulated definitions.
#[derive(Clone, Debug)]
pub struct Session {
    program: Program,
    parser: ParserState,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        let origin = Position {
            offset: 0,
            line: 1,
            column: 1,
        };
        Self {
            program: Program {
                items: Vec::new(),
                span: Span {
                    start: origin,
                    end: origin,
                },
            },
            parser: ParserState::new(),
        }
    }

    /// Add one parsed submission and evaluate only its top-level expressions.
    ///
    /// Semantic validation is transactional: an invalid submission does not
    /// alter the definitions already stored in the session.
    pub fn submit(&mut self, submission: Program) -> Result<Vec<f64>, CodegenError> {
        let first_new_item = self.program.items.len();
        let mut candidate = self.program.clone();
        candidate.span.end = submission.span.end;
        candidate.items.extend(submission.items);
        validate(&candidate)?;

        let mut values = Vec::new();
        for index in first_new_item..candidate.items.len() {
            if matches!(candidate.items[index], Item::Expression(_)) {
                values.push(compile_top_level(&candidate, index)?.call());
            }
        }

        candidate
            .items
            .retain(|item| !matches!(item, Item::Expression(_)));
        let mut parser = self.parser.clone();
        parser.install_definitions(&candidate);
        self.program = candidate;
        self.parser = parser;
        Ok(values)
    }

    /// Parse and submit source while retaining user-defined operator precedence.
    ///
    /// Parsing and semantic validation are both transactional: a rejected
    /// operator definition does not affect later submissions.
    pub fn submit_source(&mut self, source: &str) -> Result<Vec<f64>, SubmissionError> {
        let mut parser = self.parser.clone();
        let submission = parser.parse_program(source)?;
        let values = self.submit(submission)?;
        self.parser = parser;
        Ok(values)
    }

    /// The declarations and definitions retained for future submissions.
    pub fn program(&self) -> &Program {
        &self.program
    }
}

/// A syntax, semantic, or backend error from [`Session::submit_source`].
#[derive(Debug)]
pub enum SubmissionError {
    Parse(ParseError),
    Codegen(CodegenError),
}

impl fmt::Display for SubmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => error.fmt(formatter),
            Self::Codegen(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SubmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Codegen(error) => Some(error),
        }
    }
}

impl From<ParseError> for SubmissionError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl From<CodegenError> for SubmissionError {
    fn from(error: CodegenError) -> Self {
        Self::Codegen(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evaluate, parse_program};

    fn submit(session: &mut Session, source: &str) -> Result<Vec<f64>, CodegenError> {
        session.submit(parse_program(source).unwrap())
    }

    #[test]
    fn definitions_persist_between_submissions() {
        let mut session = Session::new();
        assert!(
            submit(&mut session, "def double(x) x * 2;")
                .unwrap()
                .is_empty()
        );
        assert_eq!(submit(&mut session, "double(21);").unwrap(), vec![42.0]);
        assert_eq!(submit(&mut session, "double(5);").unwrap(), vec![10.0]);
        assert_eq!(session.program().items.len(), 1);
    }

    #[test]
    fn invalid_submissions_do_not_change_the_session() {
        let mut session = Session::new();
        submit(&mut session, "def good(x) x;").unwrap();
        let item_count = session.program().items.len();
        assert!(submit(&mut session, "def bad(x) missing(x);").is_err());
        assert_eq!(session.program().items.len(), item_count);
        assert_eq!(submit(&mut session, "good(7);").unwrap(), vec![7.0]);
    }

    #[test]
    fn standard_math_externs_call_typed_host_functions() {
        let program = parse_program(
            "extern sin(x); extern cos(x); def unit(x) sin(x)*sin(x)+cos(x)*cos(x); unit(4);",
        )
        .unwrap();
        let result = evaluate(&program).unwrap()[0];
        assert!((result - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn standard_library_requires_matching_prototypes() {
        let error = evaluate(&parse_program("extern sin(x y); sin(1, 2);").unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("host binding for 'sin' expects 1 parameter"));
    }

    #[test]
    fn operators_and_precedence_persist_between_source_submissions() {
        let mut session = Session::new();
        assert!(
            session
                .submit_source("def binary@ 50 (left right) left * 10 + right;")
                .unwrap()
                .is_empty()
        );
        assert_eq!(session.submit_source("1 + 2 @ 3;").unwrap(), vec![24.0]);
    }

    #[test]
    fn rejected_operator_definitions_do_not_change_parser_state() {
        let mut session = Session::new();
        let error = session
            .submit_source("def binary@ 50 (value value) value;")
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate parameter 'value'"));

        let error = session.submit_source("1 @ 2;").unwrap_err().to_string();
        assert!(error.contains("unknown binary operator '@'"));
    }
}
