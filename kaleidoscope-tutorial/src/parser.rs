//! Chapters 2 and 6: parse source with dynamically extensible precedence.

use std::collections::HashMap;
use std::fmt;

use pest::Parser as _;
use pest::error::ErrorVariant;
use pest::iterators::Pair;

use crate::ast::{BinaryOp, Expr, ExprKind, Function, Item, Program, Prototype, PrototypeKind};
use crate::lexer::{Position, Span};
use crate::syntax::{KaleidoscopeParser, Rule};

/// A source-aware syntax error produced by Pest.
#[derive(Debug)]
pub struct ParseError(pest::error::Error<Rule>);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for ParseError {}

impl From<pest::error::Error<Rule>> for ParseError {
    fn from(error: pest::error::Error<Rule>) -> Self {
        Self(error)
    }
}

/// Stateful operator precedence used while parsing a file or REPL session.
///
/// Binary definitions extend this table in source order. Use one state for
/// successive REPL submissions so operators retain their declared precedence.
#[derive(Clone, Debug)]
pub struct ParserState {
    binary_precedence: HashMap<char, u8>,
}

impl Default for ParserState {
    fn default() -> Self {
        Self::new()
    }
}

impl ParserState {
    pub fn new() -> Self {
        Self {
            binary_precedence: HashMap::from([('<', 10), ('+', 20), ('-', 20), ('*', 40)]),
        }
    }

    /// Parse one complete file or submission, retaining successful operator
    /// definitions for the next call.
    pub fn parse_program(&mut self, source: &str) -> Result<Program, ParseError> {
        let mut candidate = self.clone();
        let program = candidate.parse_program_inner(source)?;
        *self = candidate;
        Ok(program)
    }

    pub(crate) fn install_definitions(&mut self, program: &Program) {
        for item in &program.items {
            if let Item::Definition(Function {
                prototype:
                    Prototype {
                        kind:
                            PrototypeKind::Binary {
                                operator,
                                precedence,
                            },
                        ..
                    },
                ..
            }) = item
            {
                self.binary_precedence.insert(*operator, *precedence);
            }
        }
    }

    fn parse_program_inner(&mut self, source: &str) -> Result<Program, ParseError> {
        let mut parsed = KaleidoscopeParser::parse(Rule::program, source)?;
        let program_pair = parsed
            .next()
            .expect("the program rule always produces one pair");
        let span = span_from_pest(program_pair.as_span());
        let mut items = Vec::new();

        for pair in program_pair.into_inner() {
            match pair.as_rule() {
                Rule::definition => items.push(Item::Definition(parse_definition(pair, self)?)),
                Rule::extern_declaration => items.push(Item::Extern(parse_extern(pair)?)),
                Rule::top_level_expression => {
                    let expression = pair
                        .into_inner()
                        .find(|inner| inner.as_rule() == Rule::expression)
                        .expect("a top-level expression contains an expression");
                    items.push(Item::Expression(parse_expression(expression, self)?));
                }
                Rule::empty_statement | Rule::EOI => {}
                _ => unreachable!("top-level grammar returned an unexpected rule"),
            }
        }

        Ok(Program { items, span })
    }
}

/// Parse a complete file with only the four built-in precedences installed.
pub fn parse_program(source: &str) -> Result<Program, ParseError> {
    ParserState::new().parse_program(source)
}

fn parse_definition(
    pair: Pair<'_, Rule>,
    parser: &mut ParserState,
) -> Result<Function, ParseError> {
    let span = span_from_pest(pair.as_span());
    let mut inner = pair.into_inner();
    let keyword = inner.next().expect("a definition starts with def");
    debug_assert_eq!(keyword.as_rule(), Rule::keyword_def);
    let prototype = parse_prototype(inner.next().expect("a definition has a prototype"))?;
    let body = parse_expression(inner.next().expect("a definition has a body"), parser)?;
    if let PrototypeKind::Binary {
        operator,
        precedence,
    } = prototype.kind
    {
        parser.binary_precedence.insert(operator, precedence);
    }

    Ok(Function {
        prototype,
        body,
        span,
    })
}

fn parse_extern(pair: Pair<'_, Rule>) -> Result<Prototype, ParseError> {
    let mut inner = pair.into_inner();
    let keyword = inner.next().expect("an extern starts with extern");
    debug_assert_eq!(keyword.as_rule(), Rule::keyword_extern);
    parse_prototype(inner.next().expect("an extern has a prototype"))
}

fn parse_prototype(pair: Pair<'_, Rule>) -> Result<Prototype, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::prototype);
    let span = span_from_pest(pair.as_span());
    let form = pair
        .into_inner()
        .next()
        .expect("a prototype has one source form");
    let form_span = form.as_span();
    let (name, parameters, kind) = match form.as_rule() {
        Rule::ordinary_prototype => {
            let mut inner = form.into_inner();
            let name = inner
                .next()
                .expect("an ordinary prototype has a name")
                .as_str()
                .to_owned();
            let parameters = inner.map(|pair| pair.as_str().to_owned()).collect();
            (name, parameters, PrototypeKind::Function)
        }
        Rule::unary_prototype => {
            let mut inner = form.into_inner();
            let keyword = inner.next().expect("a unary prototype starts with unary");
            debug_assert_eq!(keyword.as_rule(), Rule::keyword_unary);
            let operator =
                operator_from_pair(&inner.next().expect("a unary prototype has an operator"));
            let parameters = inner
                .map(|pair| pair.as_str().to_owned())
                .collect::<Vec<_>>();
            if parameters.len() != 1 {
                return Err(custom_error(
                    form_span,
                    "a unary operator must have exactly one operand",
                ));
            }
            (
                format!("unary{operator}"),
                parameters,
                PrototypeKind::Unary { operator },
            )
        }
        Rule::binary_prototype => {
            let mut inner = form.into_inner();
            let keyword = inner.next().expect("a binary prototype starts with binary");
            debug_assert_eq!(keyword.as_rule(), Rule::keyword_binary);
            let operator =
                operator_from_pair(&inner.next().expect("a binary prototype has an operator"));
            let mut precedence = 30;
            let mut parameters = Vec::new();
            for pair in inner {
                match pair.as_rule() {
                    Rule::number => {
                        let value = pair
                            .as_str()
                            .parse::<f64>()
                            .expect("the number grammar accepts Rust-compatible decimals");
                        if !(1.0..=100.0).contains(&value) || value.fract() != 0.0 {
                            return Err(custom_error(
                                pair.as_span(),
                                "binary precedence must be an integer from 1 through 100",
                            ));
                        }
                        precedence = value as u8;
                    }
                    Rule::identifier => parameters.push(pair.as_str().to_owned()),
                    _ => unreachable!("a binary prototype contains only precedence and names"),
                }
            }
            if parameters.len() != 2 {
                return Err(custom_error(
                    form_span,
                    "a binary operator must have exactly two operands",
                ));
            }
            (
                format!("binary{operator}"),
                parameters,
                PrototypeKind::Binary {
                    operator,
                    precedence,
                },
            )
        }
        _ => unreachable!("prototype contains a known prototype form"),
    };

    Ok(Prototype {
        name,
        parameters,
        kind,
        span,
    })
}

fn parse_expression(pair: Pair<'_, Rule>, parser: &ParserState) -> Result<Expr, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::expression);
    let mut inner = pair.into_inner();
    let first = inner
        .next()
        .expect("an expression starts with a unary expression");
    let mut values = vec![parse_unary(first, parser)?];
    let mut operators = Vec::<(BinaryOp, u8)>::new();

    while let Some(operator_pair) = inner.next() {
        let operator = operator_from_pair(&operator_pair);
        let Some(&precedence) = parser.binary_precedence.get(&operator) else {
            return Err(custom_error(
                operator_pair.as_span(),
                format!("unknown binary operator '{operator}'"),
            ));
        };
        while operators
            .last()
            .is_some_and(|(_, previous)| *previous >= precedence)
        {
            reduce_binary(&mut values, &mut operators);
        }
        operators.push((binary_op(operator), precedence));
        values.push(parse_unary(
            inner
                .next()
                .expect("a binary operator is followed by a unary expression"),
            parser,
        )?);
    }

    while !operators.is_empty() {
        reduce_binary(&mut values, &mut operators);
    }
    debug_assert_eq!(values.len(), 1);
    Ok(values.pop().expect("an expression produces one AST node"))
}

fn parse_unary(pair: Pair<'_, Rule>, parser: &ParserState) -> Result<Expr, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::unary_expression);
    let mut parts = pair.into_inner().collect::<Vec<_>>();
    let primary = parts.pop().expect("a unary expression has a primary");
    let mut operand = parse_primary(primary, parser)?;
    for operator_pair in parts.into_iter().rev() {
        let operator = operator_from_pair(&operator_pair);
        let span = Span {
            start: span_from_pest(operator_pair.as_span()).start,
            end: operand.span.end,
        };
        operand = Expr {
            kind: ExprKind::Unary {
                operator,
                operand: Box::new(operand),
            },
            span,
        };
    }
    Ok(operand)
}

fn reduce_binary(values: &mut Vec<Expr>, operators: &mut Vec<(BinaryOp, u8)>) {
    let (op, _) = operators.pop().expect("a reduction has an operator");
    let right = values.pop().expect("a binary operator has a right operand");
    let left = values.pop().expect("a binary operator has a left operand");
    let span = Span {
        start: left.span.start,
        end: right.span.end,
    };
    values.push(Expr {
        kind: ExprKind::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
        span,
    });
}

fn binary_op(operator: char) -> BinaryOp {
    match operator {
        '<' => BinaryOp::LessThan,
        '+' => BinaryOp::Add,
        '-' => BinaryOp::Subtract,
        '*' => BinaryOp::Multiply,
        operator => BinaryOp::UserDefined(operator),
    }
}

fn operator_from_pair(pair: &Pair<'_, Rule>) -> char {
    debug_assert_eq!(pair.as_rule(), Rule::operator);
    pair.as_str()
        .chars()
        .next()
        .expect("an operator contains one character")
}

fn parse_primary(pair: Pair<'_, Rule>, parser: &ParserState) -> Result<Expr, ParseError> {
    let span = span_from_pest(pair.as_span());
    match pair.as_rule() {
        Rule::number => Ok(Expr {
            kind: ExprKind::Number(
                pair.as_str()
                    .parse()
                    .expect("the number rule only accepts Rust-compatible decimals"),
            ),
            span,
        }),
        Rule::identifier_expression => parse_identifier_expression(pair, parser),
        Rule::if_expression => parse_if_expression(pair, parser),
        Rule::for_expression => parse_for_expression(pair, parser),
        Rule::parenthesized => {
            let expression = pair
                .into_inner()
                .next()
                .expect("parentheses contain an expression");
            let mut expression = parse_expression(expression, parser)?;
            expression.span = span;
            Ok(expression)
        }
        _ => unreachable!("the unary grammar only produces primary expressions"),
    }
}

fn parse_if_expression(pair: Pair<'_, Rule>, parser: &ParserState) -> Result<Expr, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::if_expression);
    let span = span_from_pest(pair.as_span());
    let mut expressions = pair
        .into_inner()
        .filter(|pair| pair.as_rule() == Rule::expression)
        .map(|pair| parse_expression(pair, parser));
    let condition = expressions.next().expect("if has a condition")?;
    let then_branch = expressions.next().expect("if has a then branch")?;
    let else_branch = expressions.next().expect("if has an else branch")?;
    debug_assert!(expressions.next().is_none());
    Ok(Expr {
        kind: ExprKind::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        span,
    })
}

fn parse_for_expression(pair: Pair<'_, Rule>, parser: &ParserState) -> Result<Expr, ParseError> {
    debug_assert_eq!(pair.as_rule(), Rule::for_expression);
    let span = span_from_pest(pair.as_span());
    let mut inner = pair.into_inner();
    let keyword = inner.next().expect("for expression starts with for");
    debug_assert_eq!(keyword.as_rule(), Rule::keyword_for);
    let variable = inner
        .next()
        .expect("for expression has an induction variable")
        .as_str()
        .to_owned();
    let mut expressions = inner
        .filter(|pair| pair.as_rule() == Rule::expression)
        .map(|pair| parse_expression(pair, parser))
        .collect::<Result<Vec<_>, _>>()?;
    let body = expressions.pop().expect("for expression has a body");
    let start = expressions.remove(0);
    let end = expressions.remove(0);
    let step = expressions.pop().map(Box::new);
    debug_assert!(expressions.is_empty());
    Ok(Expr {
        kind: ExprKind::For {
            variable,
            start: Box::new(start),
            end: Box::new(end),
            step,
            body: Box::new(body),
        },
        span,
    })
}

fn parse_identifier_expression(
    pair: Pair<'_, Rule>,
    parser: &ParserState,
) -> Result<Expr, ParseError> {
    let mut inner = pair.into_inner();
    let name_pair = inner
        .next()
        .expect("an identifier expression starts with a name");
    let name = name_pair.as_str().to_owned();
    let name_span = span_from_pest(name_pair.as_span());

    let Some(call_suffix) = inner.next() else {
        return Ok(Expr {
            kind: ExprKind::Variable(name),
            span: name_span,
        });
    };

    let span = Span {
        start: name_span.start,
        end: span_from_pest(call_suffix.as_span()).end,
    };
    let arguments = call_suffix
        .into_inner()
        .next()
        .map(|argument_list| {
            argument_list
                .into_inner()
                .map(|pair| parse_expression(pair, parser))
                .collect()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Expr {
        kind: ExprKind::Call {
            callee: name,
            arguments,
        },
        span,
    })
}

fn custom_error(span: pest::Span<'_>, message: impl Into<String>) -> ParseError {
    pest::error::Error::new_from_span(
        ErrorVariant::CustomError {
            message: message.into(),
        },
        span,
    )
    .into()
}

fn span_from_pest(span: pest::Span<'_>) -> Span {
    let start = span.start_pos();
    let end = span.end_pos();
    let (start_line, start_column) = start.line_col();
    let (end_line, end_column) = end.line_col();
    Span {
        start: Position {
            offset: start.pos(),
            line: start_line,
            column: start_column,
        },
        end: Position {
            offset: end.pos(),
            line: end_line,
            column: end_column,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_expression(source: &str) -> Expr {
        let program = parse_program(source).expect("source should parse");
        assert_eq!(program.items.len(), 1);
        match program.items.into_iter().next().unwrap() {
            Item::Expression(expression) => expression,
            item => panic!("expected an expression, found {item:?}"),
        }
    }

    fn binary(expression: &Expr) -> (BinaryOp, &Expr, &Expr) {
        match &expression.kind {
            ExprKind::Binary { op, left, right } => (*op, left, right),
            kind => panic!("expected a binary expression, found {kind:?}"),
        }
    }

    #[test]
    fn parses_literals_variables_and_calls() {
        let expression = one_expression("foo(1, x, bar())");
        let ExprKind::Call { callee, arguments } = expression.kind else {
            panic!("expected a call")
        };
        assert_eq!(callee, "foo");
        assert_eq!(arguments.len(), 3);
        assert_eq!(arguments[0].kind, ExprKind::Number(1.0));
        assert_eq!(arguments[1].kind, ExprKind::Variable("x".into()));
        assert!(matches!(arguments[2].kind, ExprKind::Call { .. }));
    }

    #[test]
    fn parses_if_then_else_expressions() {
        let expression = one_expression("if x < 3 then 1 else fib(x - 1)");
        let ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } = expression.kind
        else {
            panic!("expected an if expression")
        };
        assert_eq!(binary(&condition).0, BinaryOp::LessThan);
        assert_eq!(then_branch.kind, ExprKind::Number(1.0));
        assert!(matches!(else_branch.kind, ExprKind::Call { .. }));
    }

    #[test]
    fn parses_for_loops_with_optional_steps() {
        let expression = one_expression("for i = 1, i < n, 2 in putchard(42)");
        let ExprKind::For {
            variable,
            start,
            end,
            step,
            body,
        } = expression.kind
        else {
            panic!("expected a for expression")
        };
        assert_eq!(variable, "i");
        assert_eq!(start.kind, ExprKind::Number(1.0));
        assert_eq!(binary(&end).0, BinaryOp::LessThan);
        assert_eq!(step.unwrap().kind, ExprKind::Number(2.0));
        assert!(matches!(body.kind, ExprKind::Call { .. }));

        let ExprKind::For { step, .. } = one_expression("for i = 1, i < 2 in i").kind else {
            panic!("expected a for expression")
        };
        assert!(step.is_none());
    }

    #[test]
    fn applies_the_tutorial_precedence_table() {
        let expression = one_expression("a + b * c < d - e");
        let (op, left, right) = binary(&expression);
        assert_eq!(op, BinaryOp::LessThan);

        let (op, add_left, add_right) = binary(left);
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(add_left.kind, ExprKind::Variable("a".into()));
        assert_eq!(binary(add_right).0, BinaryOp::Multiply);

        assert_eq!(binary(right).0, BinaryOp::Subtract);
    }

    #[test]
    fn equal_precedence_is_left_associative() {
        let expression = one_expression("a - b + c");
        let (op, left, right) = binary(&expression);
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(binary(left).0, BinaryOp::Subtract);
        assert_eq!(right.kind, ExprKind::Variable("c".into()));
    }

    #[test]
    fn parentheses_override_precedence() {
        let expression = one_expression("(a + b) * c");
        let (op, left, _) = binary(&expression);
        assert_eq!(op, BinaryOp::Multiply);
        assert_eq!(binary(left).0, BinaryOp::Add);
        assert_eq!(left.span.start.column, 1);
        assert_eq!(left.span.end.column, 8);
    }

    #[test]
    fn parses_definitions_externs_and_top_level_expressions() {
        let program = parse_program(
            "# declarations\nextern sin(x);\ndef average(x y) (x + y) * .5;\naverage(3, 4);",
        )
        .unwrap();
        assert_eq!(program.items.len(), 3);

        let Item::Extern(prototype) = &program.items[0] else {
            panic!("expected extern")
        };
        assert_eq!(prototype.name, "sin");
        assert_eq!(prototype.parameters, ["x"]);

        let Item::Definition(function) = &program.items[1] else {
            panic!("expected definition")
        };
        assert_eq!(function.prototype.name, "average");
        assert_eq!(function.prototype.parameters, ["x", "y"]);
        assert_eq!(binary(&function.body).0, BinaryOp::Multiply);

        assert!(matches!(program.items[2], Item::Expression(_)));
    }

    #[test]
    fn ignores_empty_top_level_statements() {
        let program = parse_program(";;; 1; ; 2;;").unwrap();
        assert_eq!(program.items.len(), 2);
    }

    #[test]
    fn reports_syntax_errors_with_locations() {
        let error = parse_program("def broken(x  x + 1;").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("1:17"), "unexpected diagnostic: {message}");
    }

    #[test]
    fn parses_unary_and_binary_operator_prototypes() {
        let program = parse_program(
            "def unary!(value) value; def binary@ 55 (left right) right; def binary% (x y) x;",
        )
        .unwrap();
        let Item::Definition(unary) = &program.items[0] else {
            panic!("expected a unary definition")
        };
        assert_eq!(unary.prototype.name, "unary!");
        assert_eq!(unary.prototype.kind, PrototypeKind::Unary { operator: '!' });

        let Item::Definition(binary) = &program.items[1] else {
            panic!("expected a binary definition")
        };
        assert_eq!(binary.prototype.name, "binary@");
        assert_eq!(
            binary.prototype.kind,
            PrototypeKind::Binary {
                operator: '@',
                precedence: 55,
            }
        );

        let Item::Definition(default_binary) = &program.items[2] else {
            panic!("expected a binary definition")
        };
        assert_eq!(
            default_binary.prototype.kind,
            PrototypeKind::Binary {
                operator: '%',
                precedence: 30,
            }
        );
    }

    #[test]
    fn applies_declared_precedence_in_source_order() {
        let program = parse_program("def binary@ 50 (x y) y; 1 + 2 @ 3 * 4;").unwrap();
        let Item::Expression(expression) = &program.items[1] else {
            panic!("expected a top-level expression")
        };
        let (op, left, right) = binary(expression);
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(left.kind, ExprKind::Number(1.0));
        let (op, multiply_left, _) = binary(right);
        assert_eq!(op, BinaryOp::Multiply);
        assert_eq!(binary(multiply_left).0, BinaryOp::UserDefined('@'));
    }

    #[test]
    fn parser_state_retains_repl_operator_precedence() {
        let mut parser = ParserState::new();
        parser.parse_program("def binary@ 50 (x y) y;").unwrap();
        let program = parser.parse_program("1 + 2 @ 3;").unwrap();
        let Item::Expression(expression) = &program.items[0] else {
            panic!("expected an expression")
        };
        assert_eq!(binary(expression).0, BinaryOp::Add);
        assert_eq!(binary(binary(expression).2).0, BinaryOp::UserDefined('@'));
    }

    #[test]
    fn parses_chained_unary_operators_right_to_left() {
        let expression = one_expression("!!x");
        let ExprKind::Unary { operator, operand } = expression.kind else {
            panic!("expected a unary expression")
        };
        assert_eq!(operator, '!');
        assert!(matches!(
            operand.kind,
            ExprKind::Unary { operator: '!', .. }
        ));
    }

    #[test]
    fn diagnoses_invalid_operator_prototypes_and_unknown_binary_operators() {
        assert!(
            parse_program("def unary!(x y) x;")
                .unwrap_err()
                .to_string()
                .contains("exactly one operand")
        );
        assert!(
            parse_program("def binary@ 101 (x y) x;")
                .unwrap_err()
                .to_string()
                .contains("1 through 100")
        );
        assert!(
            parse_program("1 @ 2")
                .unwrap_err()
                .to_string()
                .contains("unknown binary operator '@'")
        );
        assert!(
            parse_program("def binary@ 50 (left right) left @ right;")
                .unwrap_err()
                .to_string()
                .contains("unknown binary operator '@'")
        );
    }
}
