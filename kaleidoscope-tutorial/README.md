# Kaleidoscope with `rust-lms`

This crate follows LLVM's [My First Language Frontend](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/) tutorial, replacing the hand-written C++ frontend with Rust and Pest. Starting in Chapter 3, the parsed AST will be specialized into a `rust-lms` staged computation and JIT-compiled to native code.

Only Chapter 1 is implemented so far. It introduces the language and produces a token stream. Nothing is parsed or executed yet.

## The route to native code

```text
Kaleidoscope source
        |
        v
Pest tokens                 Chapter 1 (now)
        |
        v
Rust AST                    Chapter 2
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

The important staging idea is that the source program is known while it is being compiled. Walking its AST is therefore stage-0 Rust work. The generated function retains only computations that depend on stage-1 runtime values. Chapter 1 stops before that boundary, but the lexer records source spans now so later stages do not have to recover them.

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
source = { SOI ~ token* ~ EOI }
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
$ cargo run -p kaleidoscope-tutorial -- kaleidoscope-tutorial/examples/chapter1.ks
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
$ printf 'extern sin(x); sin(.4);' | cargo run -q -p kaleidoscope-tutorial
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

The tests cover the complete Chapter 1 token vocabulary, comments, decimal spellings, keyword boundaries, empty input, and source spans.

## 1.5 Experiments

Good small changes to try before Chapter 2:

1. Permit `_` inside identifiers, then add a regression test for `_value` and `value_2`.
2. Add scientific notation such as `1.5e-3` without absorbing the `-` in ordinary expressions.
3. Change `TokenKind::Character` into named punctuation variants and compare the resulting parser design.
4. Feed malformed decimals such as `1.2.3` into the token dumper and decide whether adjacency should be rejected by the lexer or parser.

## Progress

| Chapter | Topic | Status |
| --- | --- | --- |
| 1 | Language and lexer | Implemented |
| 2 | Parser and AST | Next |
| 3 | AST specialization and native code | Planned |
| 4 | JIT and optimization | Planned |
| 5 | Control flow | Planned |
| 6 | User-defined operators | Planned |
| 7 | Mutable variables | Planned |
