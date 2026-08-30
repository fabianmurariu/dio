//! Chapter 2: parse Kaleidoscope source into the owned AST.

use std::fmt;
use std::sync::OnceLock;

use pest::Parser;
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};

use crate::ast::{BinaryOp, Expr, ExprKind, Function, Item, Program, Prototype};
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

/// Parse a complete file or REPL submission.
pub fn parse_program(source: &str) -> Result<Program, ParseError> {
    let mut parsed = KaleidoscopeParser::parse(Rule::program, source)?;
    let program_pair = parsed
        .next()
        .expect("the program rule always produces one pair");
    let span = span_from_pest(program_pair.as_span());
    let mut items = Vec::new();

    for pair in program_pair.into_inner() {
        match pair.as_rule() {
            Rule::definition => items.push(Item::Definition(parse_definition(pair))),
            Rule::extern_declaration => items.push(Item::Extern(parse_extern(pair))),
            Rule::top_level_expression => {
                let expression = pair
                    .into_inner()
                    .find(|inner| inner.as_rule() == Rule::expression)
                    .expect("a top-level expression contains an expression");
                items.push(Item::Expression(parse_expression(expression)));
            }
            Rule::empty_statement | Rule::EOI => {}
            _ => unreachable!("top-level grammar returned an unexpected rule"),
        }
    }

    Ok(Program { items, span })
}

fn parse_definition(pair: Pair<'_, Rule>) -> Function {
    let span = span_from_pest(pair.as_span());
    let mut inner = pair.into_inner();
    let keyword = inner.next().expect("a definition starts with def");
    debug_assert_eq!(keyword.as_rule(), Rule::keyword_def);
    let prototype = parse_prototype(inner.next().expect("a definition has a prototype"));
    let body = parse_expression(inner.next().expect("a definition has a body"));

    Function {
        prototype,
        body,
        span,
    }
}

fn parse_extern(pair: Pair<'_, Rule>) -> Prototype {
    let mut inner = pair.into_inner();
    let keyword = inner.next().expect("an extern starts with extern");
    debug_assert_eq!(keyword.as_rule(), Rule::keyword_extern);
    parse_prototype(inner.next().expect("an extern has a prototype"))
}

fn parse_prototype(pair: Pair<'_, Rule>) -> Prototype {
    debug_assert_eq!(pair.as_rule(), Rule::prototype);
    let span = span_from_pest(pair.as_span());
    let mut identifiers = pair.into_inner();
    let name = identifiers
        .next()
        .expect("a prototype has a function name")
        .as_str()
        .to_owned();
    let parameters = identifiers.map(|pair| pair.as_str().to_owned()).collect();

    Prototype {
        name,
        parameters,
        span,
    }
}

fn parse_expression(pair: Pair<'_, Rule>) -> Expr {
    debug_assert_eq!(pair.as_rule(), Rule::expression);
    precedence()
        .map_primary(parse_primary)
        .map_infix(|left, operator, right| {
            let op = match operator.as_rule() {
                Rule::less_than => BinaryOp::LessThan,
                Rule::add => BinaryOp::Add,
                Rule::subtract => BinaryOp::Subtract,
                Rule::multiply => BinaryOp::Multiply,
                _ => unreachable!("the Pratt parser only receives binary operators"),
            };
            let span = Span {
                start: left.span.start,
                end: right.span.end,
            };
            Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            }
        })
        .parse(pair.into_inner())
}

fn parse_primary(pair: Pair<'_, Rule>) -> Expr {
    let span = span_from_pest(pair.as_span());
    match pair.as_rule() {
        Rule::number => Expr {
            kind: ExprKind::Number(
                pair.as_str()
                    .parse()
                    .expect("the number rule only accepts Rust-compatible decimals"),
            ),
            span,
        },
        Rule::identifier_expression => parse_identifier_expression(pair),
        Rule::parenthesized => {
            let expression = pair
                .into_inner()
                .next()
                .expect("parentheses contain an expression");
            let mut expression = parse_expression(expression);
            expression.span = span;
            expression
        }
        _ => unreachable!("the Pratt parser only receives primary expressions"),
    }
}

fn parse_identifier_expression(pair: Pair<'_, Rule>) -> Expr {
    let mut inner = pair.into_inner();
    let name_pair = inner
        .next()
        .expect("an identifier expression starts with a name");
    let name = name_pair.as_str().to_owned();
    let name_span = span_from_pest(name_pair.as_span());

    let Some(call_suffix) = inner.next() else {
        return Expr {
            kind: ExprKind::Variable(name),
            span: name_span,
        };
    };

    let span = Span {
        start: name_span.start,
        end: span_from_pest(call_suffix.as_span()).end,
    };
    let arguments = call_suffix
        .into_inner()
        .next()
        .map(|argument_list| argument_list.into_inner().map(parse_expression).collect())
        .unwrap_or_default();
    Expr {
        kind: ExprKind::Call {
            callee: name,
            arguments,
        },
        span,
    }
}

fn precedence() -> &'static PrattParser<Rule> {
    static PRECEDENCE: OnceLock<PrattParser<Rule>> = OnceLock::new();
    PRECEDENCE.get_or_init(|| {
        PrattParser::new()
            .op(Op::infix(Rule::less_than, Assoc::Left))
            .op(Op::infix(Rule::add, Assoc::Left) | Op::infix(Rule::subtract, Assoc::Left))
            .op(Op::infix(Rule::multiply, Assoc::Left))
    })
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
}
