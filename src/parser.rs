use crate::ast::{Expr, Value, OrderedFloat64};
use crate::error::ParseError;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, multispace0, multispace1, one_of},
    combinator::{cut, map, map_res, opt, recognize},
    multi::many0,
    sequence::{delimited, pair, preceded, tuple},
    IResult, Offset,
};
use std::ops::Range;

/// Input type that tracks position for error reporting
pub struct Input<'a> {
    pub input: &'a str,
    pub offset: usize,
}

impl<'a> Input<'a> {
    pub fn new(input: &'a str) -> Self {
        Input { input, offset: 0 }
    }
    
    pub fn span_to(&self, end_input: &Input) -> Range<usize> {
        let start = self.offset;
        let end = end_input.offset;
        start..end
    }
}

/// Main entry point for parsing expressions
pub fn parse_expr(input: &str) -> Result<Expr, ParseError> {
    let _input_wrapper = Input::new(input);
    
    match expression(input) {
        Ok((remaining, expr)) => {
            let remaining = remaining.trim();
            if remaining.is_empty() {
                Ok(expr)
            } else {
                Err(ParseError::UnexpectedToken {
                    expected: "end of input".to_string(),
                    found: remaining.chars().next().unwrap_or(' ').to_string(),
                    span: input.len() - remaining.len()..input.len(),
                })
            }
        }
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
            Err(convert_nom_error(input, e))
        }
        Err(nom::Err::Incomplete(_)) => {
            Err(ParseError::UnexpectedEof { 
                span: input.len()..input.len() 
            })
        }
    }
}

/// Parse a complete expression (whitespace-tolerant)
fn expression(input: &str) -> IResult<&str, Expr> {
    preceded(multispace0, alt((list_expression, atom)))(input)
}

/// Parse list expressions like (+ a b) or (sum x)
fn list_expression(input: &str) -> IResult<&str, Expr> {
    delimited(
        char('('),
        preceded(
            multispace0,
            cut(alt((
                parse_add,
                parse_sub,
                parse_mul,
                parse_div,
                parse_sum,
                parse_count,
                parse_let,
            )))
        ),
        preceded(multispace0, cut(char(')')))
    )(input)
}

/// Parse atomic expressions (literals, column references)
fn atom(input: &str) -> IResult<&str, Expr> {
    alt((
        map(number, |v| Expr::Literal(v)),
        map(identifier, |s| Expr::Column(s)),
    ))(input)
}

/// Parse addition: (+ expr1 expr2 ...)
fn parse_add(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("+")(input)?;
    let (input, operands) = cut(parse_expression_list)(input)?;
    
    if operands.is_empty() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Many1,
        )));
    }
    
    Ok((input, Expr::Add(operands)))
}

/// Parse subtraction: (- expr1 expr2)
fn parse_sub(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("-")(input)?;
    let (input, operands) = cut(parse_expression_list)(input)?;
    
    if operands.len() != 2 {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Count,
        )));
    }
    
    Ok((input, Expr::Sub(
        Box::new(operands[0].clone()),
        Box::new(operands[1].clone()),
    )))
}

/// Parse multiplication: (* expr1 expr2 ...)
fn parse_mul(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("*")(input)?;
    let (input, operands) = cut(parse_expression_list)(input)?;
    
    if operands.is_empty() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Many1,
        )));
    }
    
    Ok((input, Expr::Mul(operands)))
}

/// Parse division: (/ expr1 expr2)
fn parse_div(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("/")(input)?;
    let (input, operands) = cut(parse_expression_list)(input)?;
    
    if operands.len() != 2 {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Count,
        )));
    }
    
    Ok((input, Expr::Div(
        Box::new(operands[0].clone()),
        Box::new(operands[1].clone()),
    )))
}

/// Parse sum reduction: (sum expr)
fn parse_sum(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("sum")(input)?;
    let (input, operands) = cut(parse_expression_list)(input)?;
    
    if operands.len() != 1 {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Count,
        )));
    }
    
    Ok((input, Expr::Sum(Box::new(operands[0].clone()))))
}

/// Parse count reduction: (count expr)  
fn parse_count(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("count")(input)?;
    let (input, operands) = cut(parse_expression_list)(input)?;
    
    if operands.len() != 1 {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Count,
        )));
    }
    
    Ok((input, Expr::Count(Box::new(operands[0].clone()))))
}

/// Parse let binding: (let var expr)
fn parse_let(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("let")(input)?;
    let (input, _) = cut(multispace1)(input)?;
    let (input, var_name) = cut(identifier)(input)?;
    let (input, _) = cut(multispace1)(input)?;
    let (input, expr) = cut(expression)(input)?;
    
    Ok((input, Expr::Let(var_name, Box::new(expr))))
}

/// Parse a list of expressions separated by whitespace
fn parse_expression_list(input: &str) -> IResult<&str, Vec<Expr>> {
    many0(preceded(multispace1, expression))(input)
}

/// Parse numeric literals (integers and floats)
fn number(input: &str) -> IResult<&str, Value> {
    alt((
        // Float: digits.digits or digits.digits with optional exponent
        map_res(
            recognize(tuple((
                opt(one_of("+-")),
                digit1,
                opt(tuple((char('.'), digit1))),
                opt(tuple((one_of("eE"), opt(one_of("+-")), digit1)))
            ))),
            |s: &str| {
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    s.parse::<f64>()
                        .map(|f| Value::Float64(OrderedFloat64::from(f)))
                        .map_err(|_| ())
                } else {
                    s.parse::<i64>()
                        .map(Value::Int64)
                        .map_err(|_| ())
                }
            }
        ),
        // Integer: just digits with optional sign
        map_res(
            recognize(pair(opt(one_of("+-")), digit1)),
            |s: &str| s.parse::<i64>().map(Value::Int64).map_err(|_| ())
        ),
    ))(input)
}

/// Parse identifiers (column names, variable names)
fn identifier(input: &str) -> IResult<&str, String> {
    map(
        take_while1(|c: char| c.is_alphanumeric() || c == '_'),
        |s: &str| s.to_string()
    )(input)
}

/// Convert nom parsing errors to our ParseError type
fn convert_nom_error(input: &str, error: nom::error::Error<&str>) -> ParseError {
    let offset = input.offset(error.input);
    let span = offset..offset + error.input.len().min(1);
    
    match error.code {
        nom::error::ErrorKind::Char => ParseError::UnexpectedToken {
            expected: "character".to_string(),
            found: error.input.chars().next().unwrap_or(' ').to_string(),
            span,
        },
        nom::error::ErrorKind::Tag => ParseError::UnexpectedToken {
            expected: "keyword".to_string(),
            found: error.input.split_whitespace().next().unwrap_or("").to_string(),
            span,
        },
        nom::error::ErrorKind::Many1 => ParseError::WrongArgumentCount {
            op: "operation".to_string(),
            expected: "at least 1".to_string(),
            found: 0,
            span,
        },
        nom::error::ErrorKind::Count => ParseError::WrongArgumentCount {
            op: "operation".to_string(),
            expected: "exact number".to_string(),
            found: 0, // We don't have the actual count here
            span,
        },
        _ => ParseError::NomError(format!("{:?}", error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_column() {
        let result = parse_expr("a").unwrap();
        assert_eq!(result, Expr::Column("a".to_string()));
    }

    #[test]
    fn test_parse_integer_literal() {
        let result = parse_expr("42").unwrap();
        assert_eq!(result, Expr::Literal(Value::Int64(42)));
    }

    #[test]
    fn test_parse_float_literal() {
        let result = parse_expr("3.14").unwrap();
        assert_eq!(result, Expr::Literal(Value::Float64(OrderedFloat64::from(3.14))));
    }

    #[test]
    fn test_parse_negative_numbers() {
        let result = parse_expr("-42").unwrap();
        assert_eq!(result, Expr::Literal(Value::Int64(-42)));
        
        let result = parse_expr("-3.14").unwrap();
        assert_eq!(result, Expr::Literal(Value::Float64(OrderedFloat64::from(-3.14))));
    }

    #[test]
    fn test_parse_scientific_notation() {
        let result = parse_expr("1.23e4").unwrap();
        assert_eq!(result, Expr::Literal(Value::Float64(OrderedFloat64::from(12300.0))));
        
        let result = parse_expr("1.5E-3").unwrap();
        assert_eq!(result, Expr::Literal(Value::Float64(OrderedFloat64::from(0.0015))));
    }

    #[test]
    fn test_parse_simple_addition() {
        let result = parse_expr("(+ a b)").unwrap();
        assert_eq!(result, Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
        ]));
    }

    #[test]
    fn test_parse_nary_addition() {
        let result = parse_expr("(+ a b c 42)").unwrap();
        assert_eq!(result, Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
            Expr::Column("c".to_string()),
            Expr::Literal(Value::Int64(42)),
        ]));
    }

    #[test]
    fn test_parse_subtraction() {
        let result = parse_expr("(- x y)").unwrap();
        assert_eq!(result, Expr::Sub(
            Box::new(Expr::Column("x".to_string())),
            Box::new(Expr::Column("y".to_string())),
        ));
    }

    #[test]
    fn test_parse_multiplication() {
        let result = parse_expr("(* a 2 b)").unwrap();
        assert_eq!(result, Expr::Mul(vec![
            Expr::Column("a".to_string()),
            Expr::Literal(Value::Int64(2)),
            Expr::Column("b".to_string()),
        ]));
    }

    #[test]
    fn test_parse_division() {
        let result = parse_expr("(/ x 2.0)").unwrap();
        assert_eq!(result, Expr::Div(
            Box::new(Expr::Column("x".to_string())),
            Box::new(Expr::Literal(Value::Float64(OrderedFloat64::from(2.0)))),
        ));
    }

    #[test]
    fn test_parse_sum_reduction() {
        let result = parse_expr("(sum a)").unwrap();
        assert_eq!(result, Expr::Sum(
            Box::new(Expr::Column("a".to_string()))
        ));
    }

    #[test]
    fn test_parse_count_reduction() {
        let result = parse_expr("(count x)").unwrap();
        assert_eq!(result, Expr::Count(
            Box::new(Expr::Column("x".to_string()))
        ));
    }

    #[test]
    fn test_parse_let_binding() {
        let result = parse_expr("(let tmp (+ a b))").unwrap();
        assert_eq!(result, Expr::Let(
            "tmp".to_string(),
            Box::new(Expr::Add(vec![
                Expr::Column("a".to_string()),
                Expr::Column("b".to_string()),
            ]))
        ));
    }

    #[test]
    fn test_parse_nested_expression() {
        let result = parse_expr("(+ a (* b c) (- d 1))").unwrap();
        assert_eq!(result, Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Mul(vec![
                Expr::Column("b".to_string()),
                Expr::Column("c".to_string()),
            ]),
            Expr::Sub(
                Box::new(Expr::Column("d".to_string())),
                Box::new(Expr::Literal(Value::Int64(1))),
            ),
        ]));
    }

    #[test]
    fn test_parse_complex_nested_expression() {
        let result = parse_expr("(sum (+ a (* b 2.5) (/ c (- d 1))))").unwrap();
        assert_eq!(result, Expr::Sum(
            Box::new(Expr::Add(vec![
                Expr::Column("a".to_string()),
                Expr::Mul(vec![
                    Expr::Column("b".to_string()),
                    Expr::Literal(Value::Float64(OrderedFloat64::from(2.5))),
                ]),
                Expr::Div(
                    Box::new(Expr::Column("c".to_string())),
                    Box::new(Expr::Sub(
                        Box::new(Expr::Column("d".to_string())),
                        Box::new(Expr::Literal(Value::Int64(1))),
                    )),
                ),
            ]))
        ));
    }

    #[test]
    fn test_parse_with_whitespace() {
        let result = parse_expr("  ( +   a   b   c  )  ").unwrap();
        assert_eq!(result, Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
            Expr::Column("c".to_string()),
        ]));
    }

    #[test]
    fn test_parse_error_unbalanced_parens() {
        let result = parse_expr("(+ a b");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_empty_expression() {
        let result = parse_expr("()");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_wrong_arg_count_subtraction() {
        let result = parse_expr("(- a)");
        assert!(result.is_err());
        
        let result = parse_expr("(- a b c)");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_wrong_arg_count_sum() {
        let result = parse_expr("(sum)");
        assert!(result.is_err());
        
        let result = parse_expr("(sum a b)");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_invalid_number() {
        let result = parse_expr("3.14.15");
        assert!(result.is_err());
    }
}