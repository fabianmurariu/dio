use std::ops::Range;

/// Main error type for the Dio expression evaluator
#[derive(Debug, thiserror::Error)]
pub enum DioError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("SSA conversion error: {0}")]
    SsaConversion(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Type mismatch: expected {expected}, found {found}. {context}")]
    TypeMismatch {
        expected: String,
        found: String,
        context: String,
    },
}

/// Parsing-specific errors with source location information
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    #[error("Unexpected end of input")]
    UnexpectedEof { span: Range<usize> },

    #[error("Expected '{expected}', found '{found}'")]
    UnexpectedToken {
        expected: String,
        found: String,
        span: Range<usize>,
    },

    #[error("Invalid number format: '{value}'")]
    InvalidNumber { value: String, span: Range<usize> },

    #[error("Invalid identifier: '{value}'")]
    InvalidIdentifier { value: String, span: Range<usize> },

    #[error("Unbalanced parentheses")]
    UnbalancedParens { span: Range<usize> },

    #[error("Empty expression")]
    EmptyExpression { span: Range<usize> },

    #[error("Unknown operation: '{op}'")]
    UnknownOperation { op: String, span: Range<usize> },

    #[error("Wrong number of arguments for '{op}': expected {expected}, found {found}")]
    WrongArgumentCount {
        op: String,
        expected: String,
        found: usize,
        span: Range<usize>,
    },

    #[error("Nom parsing error: {0}")]
    NomError(String),
}

impl ParseError {
    pub fn span(&self) -> Range<usize> {
        match self {
            ParseError::UnexpectedEof { span } => span.clone(),
            ParseError::UnexpectedToken { span, .. } => span.clone(),
            ParseError::InvalidNumber { span, .. } => span.clone(),
            ParseError::InvalidIdentifier { span, .. } => span.clone(),
            ParseError::UnbalancedParens { span } => span.clone(),
            ParseError::EmptyExpression { span } => span.clone(),
            ParseError::UnknownOperation { span, .. } => span.clone(),
            ParseError::WrongArgumentCount { span, .. } => span.clone(),
            ParseError::NomError(_) => 0..0,
        }
    }

    pub fn with_ariadne_report(&self, source: &str, filename: &str) -> String {
        use ariadne::{Color, Label, Report, ReportKind, Source};

        let mut output = Vec::new();

        let report = Report::build(ReportKind::Error, filename, self.span().start)
            .with_code(1001)
            .with_message(format!("{}", self))
            .with_label(
                Label::new((filename, self.span()))
                    .with_message(self.error_message())
                    .with_color(Color::Red),
            );

        let cache = (filename, Source::from(source));
        report.finish().write(cache, &mut output).unwrap();

        String::from_utf8(output).unwrap()
    }

    fn error_message(&self) -> String {
        match self {
            ParseError::UnexpectedEof { .. } => "Expression ended unexpectedly".to_string(),
            ParseError::UnexpectedToken {
                expected, found, ..
            } => format!("Expected '{}' but found '{}'", expected, found),
            ParseError::InvalidNumber { value, .. } => format!("'{}' is not a valid number", value),
            ParseError::InvalidIdentifier { value, .. } => {
                format!("'{}' is not a valid identifier", value)
            }
            ParseError::UnbalancedParens { .. } => "Missing closing parenthesis".to_string(),
            ParseError::EmptyExpression { .. } => "Expression cannot be empty".to_string(),
            ParseError::UnknownOperation { op, .. } => {
                format!("'{}' is not a recognized operation", op)
            }
            ParseError::WrongArgumentCount {
                op,
                expected,
                found,
                ..
            } => format!(
                "'{}' expects {} arguments, but {} were provided",
                op, expected, found
            ),
            ParseError::NomError(msg) => msg.clone(),
        }
    }
}

/// Convert nom errors to our ParseError type
impl From<nom::Err<nom::error::Error<&str>>> for ParseError {
    fn from(err: nom::Err<nom::error::Error<&str>>) -> Self {
        ParseError::NomError(format!("{:?}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_span() {
        let error = ParseError::UnexpectedToken {
            expected: "number".to_string(),
            found: "abc".to_string(),
            span: 5..8,
        };

        assert_eq!(error.span(), 5..8);
    }

    #[test]
    fn test_error_display() {
        let error = ParseError::InvalidNumber {
            value: "3.14.15".to_string(),
            span: 0..7,
        };

        assert_eq!(error.to_string(), "Invalid number format: '3.14.15'");
    }

    #[test]
    fn test_dio_error_from_parse_error() {
        let parse_error = ParseError::EmptyExpression { span: 0..0 };
        let dio_error = DioError::from(parse_error);

        match dio_error {
            DioError::Parse(_) => (), // Expected
            _ => panic!("Expected DioError::Parse"),
        }
    }
}
