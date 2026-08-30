//! Chapter 1: turn Kaleidoscope source text into tokens.

use std::fmt;

use pest::Parser;

use crate::syntax::{KaleidoscopeParser, Rule};

/// A one-based source position plus its zero-based UTF-8 byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    pub offset: usize,
    pub line: usize,
    pub column: usize,
}

/// The half-open source range occupied by a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

/// The tokens recognized in Chapter 1.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    Def,
    Extern,
    If,
    Then,
    Else,
    For,
    In,
    Identifier(String),
    Number(f64),
    Character(char),
    Eof,
}

/// One token together with the source range that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// A lexical error with Pest's source-aware diagnostic.
#[derive(Debug)]
pub struct LexError(pest::error::Error<Rule>);

impl fmt::Display for LexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for LexError {}

impl From<pest::error::Error<Rule>> for LexError {
    fn from(error: pest::error::Error<Rule>) -> Self {
        Self(error)
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Def => formatter.write_str("def"),
            Self::Extern => formatter.write_str("extern"),
            Self::If => formatter.write_str("if"),
            Self::Then => formatter.write_str("then"),
            Self::Else => formatter.write_str("else"),
            Self::For => formatter.write_str("for"),
            Self::In => formatter.write_str("in"),
            Self::Identifier(name) => write!(formatter, "identifier({name})"),
            Self::Number(value) => write!(formatter, "number({value})"),
            Self::Character(character) => write!(formatter, "'{character}'"),
            Self::Eof => formatter.write_str("eof"),
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}\t{}",
            self.span.start.line, self.span.start.column, self.kind
        )
    }
}

/// Lex all of `source`, including one explicit [`TokenKind::Eof`] token.
///
/// Whitespace and `#` comments are discarded by the Pest grammar. Operators,
/// parentheses, commas, and semicolons remain individual character tokens so
/// Chapter 2 can decide what they mean.
pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut parsed = KaleidoscopeParser::parse(Rule::token_stream, source)?;
    let source_pair = parsed
        .next()
        .expect("the source rule always produces a pair");
    let mut tokens = Vec::new();

    for pair in source_pair.into_inner() {
        let span = span_from_pest(pair.as_span());
        let text = pair.as_str();
        let kind = match pair.as_rule() {
            Rule::keyword_def => TokenKind::Def,
            Rule::keyword_extern => TokenKind::Extern,
            Rule::keyword_if => TokenKind::If,
            Rule::keyword_then => TokenKind::Then,
            Rule::keyword_else => TokenKind::Else,
            Rule::keyword_for => TokenKind::For,
            Rule::keyword_in => TokenKind::In,
            Rule::identifier => TokenKind::Identifier(text.to_owned()),
            Rule::number => TokenKind::Number(
                text.parse()
                    .expect("the number grammar only accepts Rust-compatible decimals"),
            ),
            Rule::character => TokenKind::Character(
                text.chars()
                    .next()
                    .expect("a character token cannot be empty"),
            ),
            Rule::EOI => continue,
            Rule::token_stream | Rule::token | Rule::WHITESPACE | Rule::COMMENT => {
                unreachable!("silent and wrapper rules are not inner tokens")
            }
            _ => unreachable!("parser-only rules do not appear in the token stream"),
        };
        tokens.push(Token { kind, span });
    }

    let end = pest::Position::new(source, source.len())
        .expect("the end of a string is always a UTF-8 boundary");
    let (line, column) = end.line_col();
    let position = Position {
        offset: source.len(),
        line,
        column,
    };
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span {
            start: position,
            end: position,
        },
    });

    Ok(tokens)
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

    fn kinds(source: &str) -> Vec<TokenKind> {
        lex(source)
            .expect("source should lex")
            .into_iter()
            .map(|token| token.kind)
            .collect()
    }

    #[test]
    fn recognizes_chapter_one_tokens() {
        assert_eq!(
            kinds("def foo(x) x + 1.0; extern sin(arg);"),
            vec![
                TokenKind::Def,
                TokenKind::Identifier("foo".into()),
                TokenKind::Character('('),
                TokenKind::Identifier("x".into()),
                TokenKind::Character(')'),
                TokenKind::Identifier("x".into()),
                TokenKind::Character('+'),
                TokenKind::Number(1.0),
                TokenKind::Character(';'),
                TokenKind::Extern,
                TokenKind::Identifier("sin".into()),
                TokenKind::Character('('),
                TokenKind::Identifier("arg".into()),
                TokenKind::Character(')'),
                TokenKind::Character(';'),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn skips_comments_and_whitespace() {
        assert_eq!(
            kinds("  # explain the answer\nanswer\t42\r\n"),
            vec![
                TokenKind::Identifier("answer".into()),
                TokenKind::Number(42.0),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn recognizes_decimal_spellings() {
        assert_eq!(
            kinds("0 12 3.5 .25 42."),
            vec![
                TokenKind::Number(0.0),
                TokenKind::Number(12.0),
                TokenKind::Number(3.5),
                TokenKind::Number(0.25),
                TokenKind::Number(42.0),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn only_exact_keywords_are_reserved() {
        assert_eq!(
            kinds(
                "def define extern external if iffy then then2 else elsewhere for format in inside"
            ),
            vec![
                TokenKind::Def,
                TokenKind::Identifier("define".into()),
                TokenKind::Extern,
                TokenKind::Identifier("external".into()),
                TokenKind::If,
                TokenKind::Identifier("iffy".into()),
                TokenKind::Then,
                TokenKind::Identifier("then2".into()),
                TokenKind::Else,
                TokenKind::Identifier("elsewhere".into()),
                TokenKind::For,
                TokenKind::Identifier("format".into()),
                TokenKind::In,
                TokenKind::Identifier("inside".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn records_source_spans() {
        let tokens = lex("# first line\ndef fib").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Def);
        assert_eq!(
            tokens[0].span.start,
            Position {
                offset: 13,
                line: 2,
                column: 1
            }
        );
        assert_eq!(
            tokens[0].span.end,
            Position {
                offset: 16,
                line: 2,
                column: 4
            }
        );
        assert_eq!(
            tokens[1].span.start,
            Position {
                offset: 17,
                line: 2,
                column: 5
            }
        );
        assert_eq!(tokens[2].kind, TokenKind::Eof);
        assert_eq!(
            tokens[2].span.start,
            Position {
                offset: 20,
                line: 2,
                column: 8
            }
        );
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }
}
