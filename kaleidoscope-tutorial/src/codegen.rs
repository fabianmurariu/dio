//! Chapter 3 and later: specialize the owned AST into typed `rust-lms` computations.

use std::collections::{HashMap, HashSet};
use std::fmt;

use rust_lms::func::{
    CompileError, CompiledFn, Ctx, FunRef0, FunRef1, FunRef2, FunRef3, FunRef4, FunRef5, FunRef6,
    FunRef7, FunRef8, FunType0, call0, call1, call2, call3, call4, call5, call6, call7, call8,
};
use rust_lms::prelude::{Compiler, Const, Var, eq, lt, not, select};

use crate::ast::{BinaryOp, Expr, ExprKind, Function, Item, Program, Prototype, PrototypeKind};
use crate::lexer::Span;
use crate::runtime::{HostFunctionRef, register_standard_externs, standard_extern_arity};

/// An owner-checked, native `fn() -> f64` produced for a top-level expression.
pub type NativeNullary = CompiledFn<FunType0<f64>>;

/// A semantic or backend error encountered after parsing.
#[derive(Debug)]
pub enum CodegenError {
    /// The source is syntactically valid but not a valid Chapter 3 program.
    Semantic { message: String, span: Span },
    /// [`compile_top_level`] was given a definition or extern item.
    NotTopLevelExpression { item_index: usize },
    /// Native code generation failed.
    Backend(CompileError),
}

impl CodegenError {
    fn semantic(message: impl Into<String>, span: Span) -> Self {
        Self::Semantic {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Semantic { message, span } => write!(
                formatter,
                "{}:{}: {message}",
                span.start.line, span.start.column
            ),
            Self::NotTopLevelExpression { item_index } => {
                write!(formatter, "item {item_index} is not a top-level expression")
            }
            Self::Backend(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CodegenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Semantic { .. } | Self::NotTopLevelExpression { .. } => None,
        }
    }
}

impl From<CompileError> for CodegenError {
    fn from(error: CompileError) -> Self {
        Self::Backend(error)
    }
}

#[derive(Clone, Copy)]
struct Signature {
    arity: usize,
    defined: bool,
}

type Ref1 = FunRef1<f64, f64>;
type Ref2 = FunRef2<f64, f64, f64>;
type Ref3 = FunRef3<f64, f64, f64, f64>;
type Ref4 = FunRef4<f64, f64, f64, f64, f64>;
type Ref5 = FunRef5<f64, f64, f64, f64, f64, f64>;
type Ref6 = FunRef6<f64, f64, f64, f64, f64, f64, f64>;
type Ref7 = FunRef7<f64, f64, f64, f64, f64, f64, f64, f64>;
type Ref8 = FunRef8<f64, f64, f64, f64, f64, f64, f64, f64, f64>;

/// A homogeneous Kaleidoscope function reference, made dynamic only in arity.
#[derive(Clone, Copy, Debug)]
enum FunctionRef {
    Zero(FunRef0<f64>),
    One(Ref1),
    Two(Ref2),
    Three(Ref3),
    Four(Ref4),
    Five(Ref5),
    Six(Ref6),
    Seven(Ref7),
    Eight(Ref8),
}

impl FunctionRef {
    fn new(arity: usize, id: usize) -> Self {
        match arity {
            0 => Self::Zero(FunRef0::new(id)),
            1 => Self::One(FunRef1::new(id)),
            2 => Self::Two(FunRef2::new(id)),
            3 => Self::Three(FunRef3::new(id)),
            4 => Self::Four(FunRef4::new(id)),
            5 => Self::Five(FunRef5::new(id)),
            6 => Self::Six(FunRef6::new(id)),
            7 => Self::Seven(FunRef7::new(id)),
            8 => Self::Eight(FunRef8::new(id)),
            _ => unreachable!("semantic validation limits functions to eight parameters"),
        }
    }

    fn id(self) -> usize {
        match self {
            Self::Zero(reference) => reference.id(),
            Self::One(reference) => reference.id(),
            Self::Two(reference) => reference.id(),
            Self::Three(reference) => reference.id(),
            Self::Four(reference) => reference.id(),
            Self::Five(reference) => reference.id(),
            Self::Six(reference) => reference.id(),
            Self::Seven(reference) => reference.id(),
            Self::Eight(reference) => reference.id(),
        }
    }

    fn emit_call(self, ctx: &mut Ctx, arguments: &[Var<f64>]) -> Var<f64> {
        match self {
            Self::Zero(reference) => ctx.bind(call0(reference)),
            Self::One(reference) => ctx.bind(call1(reference, arguments[0])),
            Self::Two(reference) => ctx.bind(call2(reference, arguments[0], arguments[1])),
            Self::Three(reference) => {
                ctx.bind(call3(reference, arguments[0], arguments[1], arguments[2]))
            }
            Self::Four(reference) => ctx.bind(call4(
                reference,
                arguments[0],
                arguments[1],
                arguments[2],
                arguments[3],
            )),
            Self::Five(reference) => ctx.bind(call5(
                reference,
                arguments[0],
                arguments[1],
                arguments[2],
                arguments[3],
                arguments[4],
            )),
            Self::Six(reference) => ctx.bind(call6(
                reference,
                arguments[0],
                arguments[1],
                arguments[2],
                arguments[3],
                arguments[4],
                arguments[5],
            )),
            Self::Seven(reference) => ctx.bind(call7(
                reference,
                arguments[0],
                arguments[1],
                arguments[2],
                arguments[3],
                arguments[4],
                arguments[5],
                arguments[6],
            )),
            Self::Eight(reference) => ctx.bind(call8(
                reference,
                arguments[0],
                arguments[1],
                arguments[2],
                arguments[3],
                arguments[4],
                arguments[5],
                arguments[6],
                arguments[7],
            )),
        }
    }
}

/// Compile one [`Item::Expression`] into an owner-checked native function.
///
/// Named definitions are compiled into the same JIT module. A fresh module is
/// used for each call, which keeps this chapter's API simple and makes the
/// returned function independently own all executable memory it can reach.
pub fn compile_top_level(
    program: &Program,
    item_index: usize,
) -> Result<NativeNullary, CodegenError> {
    validate(program)?;
    compile_top_level_validated(program, item_index)
}

/// Compile and execute every top-level expression in source order.
pub fn evaluate(program: &Program) -> Result<Vec<f64>, CodegenError> {
    validate(program)?;
    program
        .items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| matches!(item, Item::Expression(_)).then_some(index))
        .map(|index| {
            let function = compile_top_level_validated(program, index)?;
            Ok(function.call())
        })
        .collect()
}

fn compile_top_level_validated(
    program: &Program,
    item_index: usize,
) -> Result<NativeNullary, CodegenError> {
    let expression = match program.items.get(item_index) {
        Some(Item::Expression(expression)) => expression.clone(),
        _ => return Err(CodegenError::NotTopLevelExpression { item_index }),
    };

    let function_refs = predeclare_functions(program);
    let mut compiler = Compiler::new();
    let host_functions = register_standard_externs(&mut compiler);
    for item in &program.items {
        if let Item::Definition(function) = item {
            define_function(&mut compiler, function, &function_refs, &host_functions);
        }
    }

    let entry_refs = function_refs.clone();
    let entry_host_functions = host_functions.clone();
    let entry_name = format!("__rust_lms_kaleidoscope_expression_{item_index}");
    let entry = compiler.fun0(&entry_name, move |ctx| {
        lower_expr(
            &expression,
            ctx,
            &HashMap::new(),
            &entry_refs,
            &entry_host_functions,
        )
    });
    let compiled = compiler.compile(entry)?;
    Ok(compiled.as_fn())
}

/// Check names, prototypes, calls, operators, and the host environment.
pub fn validate(program: &Program) -> Result<(), CodegenError> {
    let mut signatures = HashMap::<String, Signature>::new();

    for item in &program.items {
        let (prototype, is_definition) = match item {
            Item::Definition(function) => (&function.prototype, true),
            Item::Extern(prototype) => (prototype, false),
            Item::Expression(_) => continue,
        };
        validate_prototype(prototype)?;

        match signatures.get_mut(&prototype.name) {
            Some(signature) if signature.arity != prototype.parameters.len() => {
                return Err(CodegenError::semantic(
                    format!(
                        "conflicting declaration of '{}': expected {} parameters, found {}",
                        prototype.name,
                        signature.arity,
                        prototype.parameters.len()
                    ),
                    prototype.span,
                ));
            }
            Some(signature) if is_definition && signature.defined => {
                return Err(CodegenError::semantic(
                    format!("function '{}' is defined more than once", prototype.name),
                    prototype.span,
                ));
            }
            Some(signature) => signature.defined |= is_definition,
            None => {
                signatures.insert(
                    prototype.name.clone(),
                    Signature {
                        arity: prototype.parameters.len(),
                        defined: is_definition,
                    },
                );
            }
        }
    }

    for item in &program.items {
        match item {
            Item::Definition(function) => {
                let variables = function
                    .prototype
                    .parameters
                    .iter()
                    .map(String::as_str)
                    .collect();
                validate_expr(&function.body, &variables, &signatures)?;
            }
            Item::Expression(expression) => {
                validate_expr(expression, &HashSet::new(), &signatures)?;
            }
            Item::Extern(_) => {}
        }
    }
    Ok(())
}

fn validate_prototype(prototype: &Prototype) -> Result<(), CodegenError> {
    match prototype.kind {
        PrototypeKind::Unary { .. } if prototype.parameters.len() != 1 => {
            return Err(CodegenError::semantic(
                "a unary operator must have exactly one operand",
                prototype.span,
            ));
        }
        PrototypeKind::Binary { .. } if prototype.parameters.len() != 2 => {
            return Err(CodegenError::semantic(
                "a binary operator must have exactly two operands",
                prototype.span,
            ));
        }
        PrototypeKind::Binary { precedence, .. } if !(1..=100).contains(&precedence) => {
            return Err(CodegenError::semantic(
                "binary precedence must be from 1 through 100",
                prototype.span,
            ));
        }
        PrototypeKind::Function | PrototypeKind::Unary { .. } | PrototypeKind::Binary { .. } => {}
    }

    if prototype.parameters.len() > 8 {
        return Err(CodegenError::semantic(
            format!(
                "function '{}' has {} parameters; rust-lms supports at most 8",
                prototype.name,
                prototype.parameters.len()
            ),
            prototype.span,
        ));
    }

    let mut names = HashSet::new();
    for parameter in &prototype.parameters {
        if !names.insert(parameter) {
            return Err(CodegenError::semantic(
                format!(
                    "function '{}' has duplicate parameter '{parameter}'",
                    prototype.name
                ),
                prototype.span,
            ));
        }
    }
    Ok(())
}

fn validate_expr(
    expression: &Expr,
    variables: &HashSet<&str>,
    signatures: &HashMap<String, Signature>,
) -> Result<(), CodegenError> {
    match &expression.kind {
        ExprKind::Number(_) => Ok(()),
        ExprKind::Variable(name) if variables.contains(name.as_str()) => Ok(()),
        ExprKind::Variable(name) => Err(CodegenError::semantic(
            format!("unknown variable '{name}'"),
            expression.span,
        )),
        ExprKind::Unary { operator, operand } => {
            validate_expr(operand, variables, signatures)?;
            validate_operator("unary", *operator, 1, expression.span, signatures)
        }
        ExprKind::Binary { op, left, right } => {
            validate_expr(left, variables, signatures)?;
            validate_expr(right, variables, signatures)?;
            match op {
                BinaryOp::UserDefined(operator) => {
                    validate_operator("binary", *operator, 2, expression.span, signatures)
                }
                BinaryOp::LessThan | BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => {
                    Ok(())
                }
            }
        }
        ExprKind::Call { callee, arguments } => {
            let Some(signature) = signatures.get(callee) else {
                return Err(CodegenError::semantic(
                    format!("unknown function '{callee}'"),
                    expression.span,
                ));
            };
            if signature.arity != arguments.len() {
                return Err(CodegenError::semantic(
                    format!(
                        "function '{callee}' expects {} arguments, found {}",
                        signature.arity,
                        arguments.len()
                    ),
                    expression.span,
                ));
            }
            if !signature.defined {
                match standard_extern_arity(callee) {
                    Some(host_arity) if host_arity != signature.arity => {
                        return Err(CodegenError::semantic(
                            format!(
                                "host binding for '{callee}' expects {host_arity} parameter, declaration has {}",
                                signature.arity
                            ),
                            expression.span,
                        ));
                    }
                    Some(_) => {}
                    None => {
                        return Err(CodegenError::semantic(
                            format!("external function '{callee}' has no registered host binding"),
                            expression.span,
                        ));
                    }
                }
            }
            for argument in arguments {
                validate_expr(argument, variables, signatures)?;
            }
            Ok(())
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_expr(condition, variables, signatures)?;
            validate_expr(then_branch, variables, signatures)?;
            validate_expr(else_branch, variables, signatures)
        }
        ExprKind::For {
            variable,
            start,
            end,
            step,
            body,
        } => {
            validate_expr(start, variables, signatures)?;
            let mut loop_variables = variables.clone();
            loop_variables.insert(variable);
            validate_expr(end, &loop_variables, signatures)?;
            if let Some(step) = step {
                validate_expr(step, &loop_variables, signatures)?;
            }
            validate_expr(body, &loop_variables, signatures)
        }
    }
}

fn validate_operator(
    kind: &str,
    operator: char,
    arity: usize,
    span: Span,
    signatures: &HashMap<String, Signature>,
) -> Result<(), CodegenError> {
    let name = format!("{kind}{operator}");
    let Some(signature) = signatures.get(&name) else {
        return Err(CodegenError::semantic(
            format!("unknown {kind} operator '{operator}'"),
            span,
        ));
    };
    if signature.arity != arity {
        return Err(CodegenError::semantic(
            format!("{kind} operator '{operator}' has an invalid prototype"),
            span,
        ));
    }
    if !signature.defined {
        return Err(CodegenError::semantic(
            format!("{kind} operator '{operator}' is declared but not defined"),
            span,
        ));
    }
    Ok(())
}

fn predeclare_functions(program: &Program) -> HashMap<String, FunctionRef> {
    program
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Definition(function) => Some(function),
            Item::Extern(_) | Item::Expression(_) => None,
        })
        .enumerate()
        .map(|(id, function)| {
            (
                function.prototype.name.clone(),
                FunctionRef::new(function.prototype.parameters.len(), id),
            )
        })
        .collect()
}

macro_rules! define_with_parameters {
    ($compiler:expr, $method:ident, $name:expr, $function:expr, $refs:expr, $host_functions:expr; $($argument:ident),+) => {{
        let body = $function.body.clone();
        let parameter_names = $function.prototype.parameters.clone();
        let function_refs = $refs.clone();
        let host_functions = $host_functions.clone();
        $compiler.$method($name, move |ctx, $($argument: Var<f64>),+| {
            let arguments = [$($argument),+];
            lower_function_body(
                ctx,
                &parameter_names,
                &arguments,
                &body,
                &function_refs,
                &host_functions,
            )
        })
    }};
}

fn define_function(
    compiler: &mut Compiler,
    function: &Function,
    function_refs: &HashMap<String, FunctionRef>,
    host_functions: &HashMap<String, HostFunctionRef>,
) {
    let name = &function.prototype.name;
    let actual = match function.prototype.parameters.len() {
        0 => {
            let body = function.body.clone();
            let refs = function_refs.clone();
            let hosts = host_functions.clone();
            FunctionRef::Zero(compiler.fun0(name, move |ctx| {
                lower_expr(&body, ctx, &HashMap::new(), &refs, &hosts)
            }))
        }
        1 => FunctionRef::One(define_with_parameters!(
            compiler, fun1, name, function, function_refs, host_functions; a
        )),
        2 => FunctionRef::Two(define_with_parameters!(
            compiler, fun2, name, function, function_refs, host_functions; a, b
        )),
        3 => FunctionRef::Three(define_with_parameters!(
            compiler, fun3, name, function, function_refs, host_functions; a, b, c
        )),
        4 => FunctionRef::Four(define_with_parameters!(
            compiler, fun4, name, function, function_refs, host_functions; a, b, c, d
        )),
        5 => FunctionRef::Five(define_with_parameters!(
            compiler, fun5, name, function, function_refs, host_functions; a, b, c, d, e
        )),
        6 => FunctionRef::Six(define_with_parameters!(
            compiler, fun6, name, function, function_refs, host_functions; a, b, c, d, e, f
        )),
        7 => FunctionRef::Seven(define_with_parameters!(
            compiler, fun7, name, function, function_refs, host_functions; a, b, c, d, e, f, g
        )),
        8 => FunctionRef::Eight(define_with_parameters!(
            compiler, fun8, name, function, function_refs, host_functions; a, b, c, d, e, f, g, h
        )),
        _ => unreachable!("semantic validation limits functions to eight parameters"),
    };

    debug_assert_eq!(actual.id(), function_refs[name].id());
}

fn lower_function_body(
    ctx: &mut Ctx,
    parameter_names: &[String],
    arguments: &[Var<f64>],
    body: &Expr,
    function_refs: &HashMap<String, FunctionRef>,
    host_functions: &HashMap<String, HostFunctionRef>,
) -> Var<f64> {
    let variables = parameter_names
        .iter()
        .cloned()
        .zip(arguments.iter().copied())
        .collect();
    lower_expr(body, ctx, &variables, function_refs, host_functions)
}

fn lower_expr(
    expression: &Expr,
    ctx: &mut Ctx,
    variables: &HashMap<String, Var<f64>>,
    function_refs: &HashMap<String, FunctionRef>,
    host_functions: &HashMap<String, HostFunctionRef>,
) -> Var<f64> {
    match &expression.kind {
        ExprKind::Number(value) => ctx.bind(Const::<f64>::new(*value)),
        ExprKind::Variable(name) => variables[name],
        ExprKind::Unary { operator, operand } => {
            let operand = lower_expr(operand, ctx, variables, function_refs, host_functions);
            function_refs[&format!("unary{operator}")].emit_call(ctx, &[operand])
        }
        ExprKind::Binary { op, left, right } => {
            let left = lower_expr(left, ctx, variables, function_refs, host_functions);
            let right = lower_expr(right, ctx, variables, function_refs, host_functions);
            match op {
                BinaryOp::LessThan => ctx.bind(select(lt(left, right), 1.0f64, 0.0f64)),
                BinaryOp::Add => ctx.bind(left + right),
                BinaryOp::Subtract => ctx.bind(left - right),
                BinaryOp::Multiply => ctx.bind(left * right),
                BinaryOp::UserDefined(operator) => {
                    function_refs[&format!("binary{operator}")].emit_call(ctx, &[left, right])
                }
            }
        }
        ExprKind::Call { callee, arguments } => {
            let arguments = arguments
                .iter()
                .map(|argument| lower_expr(argument, ctx, variables, function_refs, host_functions))
                .collect::<Vec<_>>();
            if let Some(function) = function_refs.get(callee) {
                function.emit_call(ctx, &arguments)
            } else {
                host_functions[callee].emit_call(ctx, arguments[0])
            }
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = lower_expr(condition, ctx, variables, function_refs, host_functions);
            let result = ctx.var(0.0f64);
            ctx.if_then_else(
                not(eq(condition, 0.0f64)),
                |then_ctx| {
                    let value = lower_expr(
                        then_branch,
                        then_ctx,
                        variables,
                        function_refs,
                        host_functions,
                    );
                    then_ctx.store(result, value);
                },
                |else_ctx| {
                    let value = lower_expr(
                        else_branch,
                        else_ctx,
                        variables,
                        function_refs,
                        host_functions,
                    );
                    else_ctx.store(result, value);
                },
            );
            result
        }
        ExprKind::For {
            variable,
            start,
            end,
            step,
            body,
        } => {
            let start = lower_expr(start, ctx, variables, function_refs, host_functions);
            let induction = ctx.var(start);
            let mut loop_variables = variables.clone();
            loop_variables.insert(variable.clone(), induction);

            ctx.while_loop(true, |loop_ctx| {
                let _body_value = lower_expr(
                    body,
                    loop_ctx,
                    &loop_variables,
                    function_refs,
                    host_functions,
                );
                let step = match step {
                    Some(step) => lower_expr(
                        step,
                        loop_ctx,
                        &loop_variables,
                        function_refs,
                        host_functions,
                    ),
                    None => loop_ctx.bind(Const::<f64>::new(1.0)),
                };
                let end = lower_expr(
                    end,
                    loop_ctx,
                    &loop_variables,
                    function_refs,
                    host_functions,
                );
                loop_ctx.store(induction, induction + step);
                loop_ctx.if_then(eq(end, 0.0f64), |break_ctx| break_ctx.break_loop());
            });
            ctx.bind(Const::<f64>::new(0.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_program;

    fn values(source: &str) -> Result<Vec<f64>, CodegenError> {
        evaluate(&parse_program(source).expect("source should parse"))
    }

    fn error(source: &str) -> String {
        values(source)
            .expect_err("program should be rejected")
            .to_string()
    }

    #[test]
    fn compiles_arithmetic_with_parser_precedence() {
        assert_eq!(values("1 + 2 * 3").unwrap(), vec![7.0]);
    }

    #[test]
    fn compiles_named_functions_and_calls() {
        let source = "def square(x) x * x; def sumsq(x y) square(x) + square(y); sumsq(3, 4);";
        assert_eq!(values(source).unwrap(), vec![25.0]);
    }

    #[test]
    fn comparison_produces_kaleidoscope_numbers() {
        assert_eq!(values("2 < 3; 3 < 2;").unwrap(), vec![1.0, 0.0]);
    }

    #[test]
    fn compiles_value_producing_conditionals() {
        assert_eq!(values("if 1 then 11 else 22").unwrap(), vec![11.0]);
        assert_eq!(values("if 0 then 11 else 22").unwrap(), vec![22.0]);
        assert_eq!(values("if 2 then 11 else 22").unwrap(), vec![11.0]);
        assert_eq!(
            values("1 + if 1 then if 0 then 2 else 3 else 4").unwrap(),
            vec![4.0]
        );
    }

    #[test]
    fn executes_only_the_selected_branch() {
        let source = "def forever() forever(); if 1 then 42 else forever();";
        assert_eq!(values(source).unwrap(), vec![42.0]);
    }

    #[test]
    fn compiles_recursive_fibonacci() {
        let source = "def fib(x) if x < 3 then 1 else fib(x-1) + fib(x-2); fib(10);";
        assert_eq!(values(source).unwrap(), vec![55.0]);
    }

    #[test]
    fn compiles_for_loops_and_the_default_step() {
        assert_eq!(values("for i = 1, i < 4 in i;").unwrap(), vec![0.0]);
        assert_eq!(values("for i = 1, i < 5, 2 in i;").unwrap(), vec![0.0]);
    }

    #[test]
    fn loop_variables_shadow_but_do_not_escape() {
        assert_eq!(
            values("def f(i) (for i = 1, i < 2 in i) + i; f(9);").unwrap(),
            vec![9.0]
        );
        assert!(error("(for i = 1, i < 2 in i) + i").contains("unknown variable 'i'"));
    }

    #[test]
    fn supports_forward_function_references() {
        let source = "def twice(x) add(x, x); def add(x y) x + y; twice(21);";
        assert_eq!(values(source).unwrap(), vec![42.0]);
    }

    #[test]
    fn stages_recursive_references_before_control_flow_exists() {
        let source = "def forever(x) forever(x); 42;";
        assert_eq!(values(source).unwrap(), vec![42.0]);
    }

    #[test]
    fn dispatches_zero_and_eight_argument_functions() {
        let source = "
            def zero() 0;
            def sum8(a b c d e f g h) a + b + c + d + e + f + g + h;
            zero();
            sum8(1, 2, 3, 4, 5, 6, 7, 8);
        ";
        assert_eq!(values(source).unwrap(), vec![0.0, 36.0]);
    }

    #[test]
    fn reports_unknown_variables() {
        assert!(error("def bad(x) x + y; 1;").contains("unknown variable 'y'"));
    }

    #[test]
    fn reports_unknown_functions() {
        assert!(error("missing(1)").contains("unknown function 'missing'"));
    }

    #[test]
    fn reports_call_arity_mismatches() {
        assert!(
            error("def add(x y) x + y; add(1);")
                .contains("function 'add' expects 2 arguments, found 1")
        );
    }

    #[test]
    fn reports_unregistered_externals() {
        assert!(
            error("extern mystery(x); mystery(1);")
                .contains("external function 'mystery' has no registered host binding")
        );
    }

    #[test]
    fn rejects_conflicting_and_duplicate_definitions() {
        assert!(error("extern f(x); def f(x y) x + y;").contains("conflicting declaration of 'f'"));
        assert!(
            error("def f(x) x; def f(y) y;").contains("function 'f' is defined more than once")
        );
    }

    #[test]
    fn rejects_duplicate_parameters_and_unsupported_arity() {
        assert!(error("def f(x x) x;").contains("duplicate parameter 'x'"));
        assert!(error("extern f(a b c d e f g h i);").contains("rust-lms supports at most 8"));
    }

    #[test]
    fn returns_an_owner_checked_native_function() {
        let program = parse_program("40 + 2").unwrap();
        let function = compile_top_level(&program, 0).unwrap();
        assert_eq!(function.call(), 42.0);
    }

    #[test]
    fn compiles_user_defined_unary_and_binary_operators() {
        let source = "
            def unary!(value) if value then 0 else 1;
            def binary> 10 (left right) right < left;
            def binary| 5 (left right)
                if left then 1 else if right then 1 else 0;
            def binary= 9 (left right) !(left < right | left > right);
            !0; !42; 3 > 2; 2 > 3; 3 = 3; 3 = 4;
        ";
        assert_eq!(values(source).unwrap(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn user_defined_binary_operators_are_left_associative_calls() {
        let source = "def binary: 1 (left right) left * 10 + right; 1 : 2 : 3;";
        assert_eq!(values(source).unwrap(), vec![123.0]);
    }

    #[test]
    fn reports_unknown_unary_operators() {
        assert!(error("!1").contains("unknown unary operator '!'"));
    }
}
