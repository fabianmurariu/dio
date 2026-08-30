# Kaleidoscope with `rust-lms`

This crate follows LLVM's [My First Language Frontend](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/) tutorial, replacing the hand-written C++ frontend with Rust and Pest. Starting in Chapter 3, the parsed AST is specialized into a `rust-lms` staged computation and JIT-compiled to native code.

Chapters 1 through 3 are implemented. The crate can tokenize a source file, parse it into an owned source-spanned AST, and compile top-level expressions and user-defined functions to native code.

## The route to native code

```text
Kaleidoscope source
        |
        v
Pest tokens                 Chapter 1
        |
        v
Rust AST                    Chapter 2
        |
        v
rust-lms staged program     Chapter 3 (now)
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

At this point `Program` is a complete static description of the Chapter 2 language. Chapter 3 walks it at stage 0, translates expressions into `rust-lms` staged values, and executes the resulting native code.

# Chapter 3: AST specialization and native code

LLVM's Chapter 3 gives each AST node a `codegen()` method which constructs LLVM IR. This implementation keeps the AST independent of any backend and puts the corresponding traversal in [`src/codegen.rs`](src/codegen.rs). The traversal constructs typed `rust-lms` computations, and `rust-lms` performs the lower-level IR construction and JIT compilation.

That difference is the point of this version of the tutorial. We still decide what each source construct means, but do not manually choose IR instructions, basic blocks, function signatures, or return instructions for straight-line expressions.

## 3.1 The two stages

Consider:

```text
def square(x) x * x;
square(4);
```

There are two distinct execution times:

```text
stage 0: parse and walk the known AST
         Number(4)       -> Const<f64>
         Variable("x")   -> Var<f64>
         Multiply        -> staged left * right
         Call("square")  -> staged call1(...)
                              |
                              v
stage 1: execute the resulting native f64 computation
```

The `match` over `ExprKind`, hash-table lookup by variable name, recursive AST traversal, and dynamic function-arity dispatch all happen at stage 0. They do not appear in the native function. Only operations depending on runtime values remain at stage 1. In partial-evaluation terminology, the AST is static input and the native computation is the residual program.

This is the central advantage of using `rust-lms` here: [`lower_expr`](src/codegen.rs) is ordinary Rust over an ordinary owned AST, while its result is a typed `Var<f64>` describing a future value.

## 3.2 Lowering expressions

Every Kaleidoscope value has the same source-language type, `f64`, so every lowering branch can return `Var<f64>`:

| Kaleidoscope AST | `rust-lms` staged computation |
| --- | --- |
| number | `ctx.bind(Const::<f64>::new(value))` |
| parameter reference | the parameter's `Var<f64>` |
| `left + right` | `ctx.bind(left + right)` |
| `left - right` | `ctx.bind(left - right)` |
| `left * right` | `ctx.bind(left * right)` |
| `left < right` | `ctx.bind(select(lt(left, right), 1.0, 0.0))` |
| function call | `ctx.bind(callN(function, ...))` |

The comparison deserves attention. `lt` has staged type `bool`, but Kaleidoscope represents truth as the number `1.0` and falsehood as `0.0`. A staged `select` converts the boolean result back to the language's single value type. It is the `rust-lms` equivalent of LLVM's floating comparison followed by an unsigned-integer-to-floating conversion.

`Ctx::bind` names the result of a staged operation. This makes every recursively lowered expression uniform and ensures a nested operation is emitted once before its result is reused.

## 3.3 Variables and semantic validation

Parameters arrive in a staged function body as typed `Var<f64>` values. The lowering pass zips those values with the prototype's parameter names:

```text
def distanceSquared(x y) x * x + y * y
                    | |   |       |
                    | `---+-------+-- Var<f64> for y
                    `-----+---------- Var<f64> for x
```

A stage-0 `HashMap<String, Var<f64>>` resolves variable expressions. This symbol table is a compiler data structure, not a runtime hash table.

Before staging begins, the crate validates the complete program. It reports source-positioned errors for:

- unknown variables and functions;
- conflicting declarations and duplicate definitions;
- duplicate parameter names;
- call argument-count mismatches;
- functions exceeding the current eight-parameter `rust-lms` API;
- a call to an `extern` declaration with no host binding.

Separating validation from lowering is useful because a `rust-lms` function body must produce a staged value, not a recoverable `Result`. Once validation succeeds, lookups in the lowering pass are guaranteed to exist.

## 3.4 Function references and forward calls

`rust-lms` gives every arity its own Rust type: `FunRef0<f64>`, `FunRef1<f64, f64>`, and so on through `FunRef8`. Kaleidoscope discovers arity only after parsing, so [`src/codegen.rs`](src/codegen.rs) uses a small `FunctionRef` enum whose variants hold those statically typed references.

All definitions are predeclared with their eventual compiler IDs before any body is staged. Definitions are then added to a fresh `Compiler` in exactly that order. As a result, this works even though `add` appears later in the file:

```text
def twice(x) add(x, x);
def add(x y) x + y;
twice(21);
```

It also prepares the representation for recursion. Actual conditional recursion becomes useful in Chapter 5, after the language gains `if` expressions.

## 3.5 Compiling a top-level expression

The public API is:

```rust
use kaleidoscope_tutorial::{compile_top_level, parse_program};

let program = parse_program("def square(x) x * x; square(9);")?;
let square_nine = compile_top_level(&program, 1)?;
assert_eq!(square_nine.call(), 81.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A top-level expression has no parameters, so it is lowered inside a generated `fun0`. `Compiler::compile` turns that function and the program's named definitions into native code. Calling `as_fn()` returns `CompiledFn<FunType0<f64>>`, which shares ownership of the executable allocation. The function therefore cannot outlive its machine code through this safe API.

`evaluate(&program)` repeats that process for every top-level expression in source order and calls each resulting function. This batch-oriented implementation deliberately uses a fresh module per expression. Chapter 4 will introduce the more interactive JIT model and external host functions.

## 3.6 Run it and inspect the generated IR

The Chapter 3 example contains two definitions and one top-level call:

```console
$ cargo run -q -p kaleidoscope-tutorial -- --run kaleidoscope-tutorial/examples/chapter3.ks
Evaluated to 25.000000
```

The older modes remain available:

```console
$ cargo run -q -p kaleidoscope-tutorial -- --ast kaleidoscope-tutorial/examples/chapter3.ks
$ cargo run -q -p kaleidoscope-tutorial -- --tokens kaleidoscope-tutorial/examples/chapter3.ks
```

Set `RUST_LMS_DEBUG_IR` to inspect the Cranelift IR emitted below the staging layer:

```console
$ RUST_LMS_DEBUG_IR=1 cargo run -q -p kaleidoscope-tutorial -- --run kaleidoscope-tutorial/examples/chapter3.ks
=== Function: square ===
...
=== Function: sumOfSquares ===
...
=== Function: __rust_lms_kaleidoscope_expression_2 ===
...
Evaluated to 25.000000
```

Notice what is absent from those functions: there is no AST interpreter, string lookup, `ExprKind` tag, or arity enum. Partial evaluation used all of those static structures while constructing the staged program.

## 3.7 Current boundaries

Chapter 3 intentionally has three visible boundaries:

1. `extern` prototypes can be declared, but calling one reports that it has no host binding. Chapter 4 will connect selected declarations to safe `rust-lms` FFI handles.
2. `rust-lms` currently exposes staged functions from arity zero through eight, so the semantic checker gives larger prototypes a clear error.
3. Each top-level expression currently owns a fresh JIT module containing the program's definitions. This is simple and lifetime-safe, but recompiles those definitions. Chapter 4 will address the interactive compilation model and optimization workflow.

The tests in `codegen.rs` cover arithmetic, precedence, numeric truth values, named calls, forward references, owner-checked execution, and every semantic error category above.

## Progress

| Chapter | Topic | Status |
| --- | --- | --- |
| 1 | Language and lexer | Implemented |
| 2 | Parser and AST | Implemented |
| 3 | AST specialization and native code | Implemented |
| 4 | JIT, host externs, and optimization | Next |
| 5 | Control flow | Planned |
| 6 | User-defined operators | Planned |
| 7 | Mutable variables | Planned |
