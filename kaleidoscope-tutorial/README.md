# Kaleidoscope with `rust-lms`

This crate follows LLVM's [My First Language Frontend](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/) tutorial, replacing the hand-written C++ frontend with Rust and Pest. Starting in Chapter 3, the parsed AST will be specialized into a `rust-lms` staged computation and JIT-compiled to native code.

Chapters 1 and 2 are implemented. The crate can tokenize a source file and parse it into an owned, source-spanned AST. Nothing is executed yet.

## The route to native code

```text
Kaleidoscope source
        |
        v
Pest tokens                 Chapter 1
        |
        v
Rust AST                    Chapter 2 (now)
        |
        v
rust-lms staged program     Chapter 3
        |
        v
Cranelift / LLVM JIT        Chapters 3–4
        |
        v
native machine code
```

The important staging idea is that the source program is known while it is being compiled. Walking its AST is therefore stage-0 Rust work. The generated function retains only computations that depend on stage-1 runtime values. Chapters 1 and 2 stop before that boundary, but both tokens and AST nodes record source spans now so later stages do not have to recover them.

# Chapter 1: the language and lexer

## 1.1 The initial language

Kaleidoscope deliberately has one value type: an IEEE-754 64-bit floating-point number, represented by Rust's `f64`. Function parameters therefore need names but not type annotations:

```text
def average(x y)
  (x + y) * .5;
```

Functions can be declared before their implementation is supplied by the host environment:

```text
extern sin(arg);
sin(.4);
```

The first chapter recognizes six kinds of token:

| Source | Token |
| --- | --- |
| `def` | `Def` |
| `extern` | `Extern` |
| `average`, `x`, `sin` | `Identifier(String)` |
| `1`, `3.14`, `.5`, `42.` | `Number(f64)` |
| `+`, `-`, `(`, `)`, `;` | `Character(char)` |
| end of input | `Eof` |

Whitespace and comments beginning with `#` are discarded. Punctuation remains a character token because the parser introduced in Chapter 2 will decide whether a character is an operator, delimiter, or terminator.

## 1.2 Describing tokens with Pest

The grammar lives in [`src/kaleidoscope.pest`](src/kaleidoscope.pest). Its core rules are:

```pest
token_stream = { SOI ~ token* ~ EOI }
token = _{ keyword_def | keyword_extern | number | identifier | character }

identifier = @{ ASCII_ALPHA ~ ASCII_ALPHANUMERIC* }
number = @{ (ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT*)?) | ("." ~ ASCII_DIGIT+) }
character = @{ ANY }
```

Pest is a parsing-expression-grammar library rather than a traditional generated lexer. Ordered alternatives matter: keywords appear before identifiers, and the catch-all character rule appears last. The keyword rules also check their boundary, so `def` is reserved while `define` remains an identifier.

The original LLVM tutorial scans numbers with a permissive `[0-9.]+` loop. Here the grammar spells out decimal syntax, permitting at most one decimal point. A leading sign is intentionally not part of a number: `-1` becomes `'-'` followed by `Number(1.0)`, allowing the parser to give unary minus its language-defined meaning later.

## 1.3 Turning Pest pairs into owned tokens

[`src/lexer.rs`](src/lexer.rs) exposes:

```rust
pub fn lex(source: &str) -> Result<Vec<Token>, LexError>
```

Each Pest match is converted into an owned `TokenKind`. Identifiers own their text and numbers are converted immediately to `f64`. Tokens also retain a half-open `Span` containing byte offsets and one-based line and column positions:

```rust
use kaleidoscope_tutorial::{TokenKind, lex};

let tokens = lex("def answer() 42;")?;
assert_eq!(tokens[0].kind, TokenKind::Def);
assert_eq!(tokens[0].span.start.line, 1);
assert_eq!(tokens[0].span.start.column, 1);
# Ok::<(), kaleidoscope_tutorial::LexError>(())
```

Keeping tokens owned makes the future AST straightforward to store, while retaining spans prepares us for useful syntax errors in Chapter 2.

## 1.4 Run it

From the workspace root, dump the tokens in the included example:

```console
$ cargo run -p kaleidoscope-tutorial -- --tokens kaleidoscope-tutorial/examples/chapter1.ks
2:1     def
2:5     identifier(average)
2:12    '('
2:13    identifier(x)
...
6:19    ';'
7:1     eof
```

With no path, the program reads standard input:

```console
$ printf 'extern sin(x); sin(.4);' | cargo run -q -p kaleidoscope-tutorial -- --tokens
1:1     extern
1:8     identifier(sin)
1:11    '('
...
1:25    eof
```

Run the chapter's tests with:

```console
$ cargo test -p kaleidoscope-tutorial
```

The lexer tests cover the complete Chapter 1 token vocabulary, comments, decimal spellings, keyword boundaries, empty input, and source spans.

## 1.5 Experiments

Good small changes to try before Chapter 2:

1. Permit `_` inside identifiers, then add a regression test for `_value` and `value_2`.
2. Add scientific notation such as `1.5e-3` without absorbing the `-` in ordinary expressions.
3. Change `TokenKind::Character` into named punctuation variants and compare the resulting parser design.
4. Feed malformed decimals such as `1.2.3` into the token dumper and decide whether adjacency should be rejected by the lexer or parser.

# Chapter 2: parser and AST

LLVM's implementation combines recursive descent with a hand-written operator-precedence parser. Pest already handles the recursive grammar, and its `PrattParser` provides the same precedence-climbing behavior for binary expressions. The result is still the same language and essentially the same AST.

## 2.1 An AST that does not know about Pest

The types in [`src/ast.rs`](src/ast.rs) contain ordinary owned Rust data:

```rust
pub enum ExprKind {
    Number(f64),
    Variable(String),
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        callee: String,
        arguments: Vec<Expr>,
    },
}
```

The remaining nodes describe function interfaces and top-level constructs:

```text
Program
`-- Vec<Item>
    |-- Definition(Function { Prototype, body: Expr })
    |-- Extern(Prototype)
    `-- Expression(Expr)
```

Keeping `ExprKind` separate from `Expr` lets every expression carry one `Span` without repeating a span field in every variant. The AST deliberately contains no Pest pairs and no `rust-lms` values. This separation gives Chapter 3 a clean input and keeps parser lifetimes out of code generation.

The LLVM tutorial immediately wraps a top-level expression in an anonymous `__anon_expr` function. This implementation retains `Item::Expression` for now; Chapter 3 will create that staged nullary function at the point where it is actually needed.

## 2.2 Grammar for complete programs

The Chapter 2 rules extend the same [`src/kaleidoscope.pest`](src/kaleidoscope.pest) file:

```pest
program = { SOI ~ top_level* ~ EOI }
top_level = _{ definition | extern_declaration | top_level_expression | empty_statement }

definition = { keyword_def ~ prototype ~ expression ~ ";"? }
extern_declaration = { keyword_extern ~ prototype ~ ";"? }
prototype = { identifier ~ "(" ~ identifier* ~ ")" }

expression = { primary ~ (binary_operator ~ primary)* }
primary = _{ number | parenthesized | identifier_expression }
```

As in the original language, parameters in a prototype are separated by whitespace:

```text
def average(x y) (x + y) * .5
```

Arguments at a call site use commas:

```text
average(3, 4)
```

Semicolons are accepted after definitions, externs, and expressions, and otherwise ignored at the top level. This supports both tutorial files and the semicolon-oriented interactive style.

## 2.3 Variables versus calls

An identifier is a variable unless it is followed by a call suffix:

```pest
identifier_expression = { identifier ~ call_suffix? }
call_suffix = { "(" ~ argument_list? ~ ")" }
argument_list = { expression ~ ("," ~ expression)* }
```

Consequently, the parser turns `x` into `ExprKind::Variable("x")`, `foo()` into a zero-argument call, and `foo(x, 1 + y)` into a call containing two recursively parsed expressions.

## 2.4 Binary precedence with a Pratt parser

Chapter 2 defines this table, from weakest to strongest binding:

| Operators | Precedence | Associativity |
| --- | --- | --- |
| `<` | 10 | left |
| `+`, `-` | 20 | left |
| `*` | 40 | left |

[`src/parser.rs`](src/parser.rs) expresses the same ordering with Pest's Pratt parser:

```rust
PrattParser::new()
    .op(Op::infix(Rule::less_than, Assoc::Left))
    .op(Op::infix(Rule::add, Assoc::Left)
        | Op::infix(Rule::subtract, Assoc::Left))
    .op(Op::infix(Rule::multiply, Assoc::Left))
```

Thus:

```text
a + b * c < d - e
```

becomes conceptually:

```text
LessThan
|-- Add
|   |-- a
|   `-- Multiply(b, c)
`-- Subtract(d, e)
```

Parentheses are primary expressions, so `(a + b) * c` naturally changes the tree without requiring a parenthesis AST node.

## 2.5 Parse a program

The public entry point is:

```rust
use kaleidoscope_tutorial::{Item, parse_program};

let program = parse_program("def square(x) x * x; square(4);")?;
assert!(matches!(program.items[0], Item::Definition(_)));
assert!(matches!(program.items[1], Item::Expression(_)));
# Ok::<(), kaleidoscope_tutorial::ParseError>(())
```

Run the included Chapter 2 program to print its AST:

```console
$ cargo run -p kaleidoscope-tutorial -- kaleidoscope-tutorial/examples/chapter2.ks
Program {
    items: [
        Extern(
            Prototype {
                name: "sin",
                ...
            },
        ),
        Definition(
            ...
        ),
        Expression(
            ...
        ),
    ],
    ...
}
```

The default CLI mode parses and prints the AST. Pass `--tokens` to revisit the Chapter 1 token stream. Both modes accept one source path or read standard input when no path is supplied.

Malformed input produces Pest's source-oriented diagnostic. For example, a missing `)` reports the line, column, expected grammar rule, and marked source line.

## 2.6 What the tests establish

The parser tests verify:

- literals, variables, nested calls, and zero-argument calls;
- the complete precedence table and left associativity;
- parentheses overriding precedence;
- definitions, extern prototypes, comments, and top-level expressions;
- ignored top-level semicolons;
- source-spanned syntax errors.

At this point `Program` is a complete static description of the Chapter 2 language. Chapter 3 will walk it at stage 0, translate expressions into `rust-lms` staged values, and execute the resulting native code.

## Progress

| Chapter | Topic | Status |
| --- | --- | --- |
| 1 | Language and lexer | Implemented |
| 2 | Parser and AST | Implemented |
| 3 | AST specialization and native code | Next |
| 4 | JIT and optimization | Planned |
| 5 | Control flow | Planned |
| 6 | User-defined operators | Planned |
| 7 | Mutable variables | Planned |
