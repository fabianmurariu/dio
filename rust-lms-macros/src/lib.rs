use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Block, Pat, Stmt};

/// Sequence-oriented block macro for staged computations.
///
/// Transforms a block of staged statements into the correct tuple sequence
/// without requiring manual threading of `LetVar` initializers.
///
/// # Transformation rules
///
/// | Input form              | Output                                                         |
/// |-------------------------|----------------------------------------------------------------|
/// | `let x = ctx.XXX(…);`  | `let __sl_x = ctx.XXX(…); let x = __sl_x.var();` + `__sl_x` in the tuple |
/// | `stmt_expr;`            | `stmt_expr` at this position in the tuple                      |
/// | `final_expr` (no `;`)   | last element of the tuple (the return value)                   |
///
/// For `let x = ctx.let_var(…)` and `let x = ctx.bind(…)`, the macro
/// automatically generates an internal `__sl_x: LetVar<T, _>` that sits in the
/// sequence tuple (triggering initialization), while `x` becomes a `Var<T>`
/// that is usable directly — **no `*` deref required**.
///
/// All other `let` bindings (non-`ctx` calls) follow the same split pattern and
/// require `T: Copy` for `.var()` to work; this holds for all primitive staged
/// types (`I64Type`, `U64Type`, `F64Type`, `BoolType`, `UnitType`).
///
/// # Example
///
/// ```ignore
/// let count_to_n = compiler.fun1("count_to_n", |ctx, n: Var<I64Type>| {
///     staged_block! {
///         let i   = ctx.let_var(0i64);
///         let sum = ctx.let_var(0i64);
///         while_loop(lt(i, n), staged_block! {
///             assign(sum, add(sum, i));
///             assign(i,   add(i, 1i64));
///         });
///         sum
///     }
/// });
/// ```
#[proc_macro]
pub fn staged_block(input: TokenStream) -> TokenStream {
    let stmts = parse_macro_input!(input with Block::parse_within);
    expand(stmts).into()
}

fn expand(stmts: Vec<Stmt>) -> TokenStream2 {
    if stmts.is_empty() {
        return quote! { () };
    }

    // let_decls: pure Rust bindings emitted before the tuple
    // seq_elems: elements that appear inside the tuple (in order)
    let mut let_decls: Vec<TokenStream2> = vec![];
    let mut seq_elems: Vec<TokenStream2> = vec![];

    for stmt in stmts {
        match stmt {
            Stmt::Local(local) => {
                let Some(ref init) = local.init else {
                    return syn::Error::new_spanned(
                        &local.let_token,
                        "staged_block! let bindings must have an initializer",
                    )
                    .to_compile_error();
                };
                let init_expr = &init.expr;
                let pat = &local.pat;

                match extract_ident(pat) {
                    Ok(ident) => {
                        // Internal name for the LetVar (goes into the sequence tuple)
                        let lv = format_ident!("__sl_{}", ident);
                        // User-facing name becomes Var<T> via .var() (no * needed)
                        let_decls.push(quote! {
                            let #lv = #init_expr;
                            let #ident = #lv.var();
                        });
                        seq_elems.push(quote! { #lv });
                    }
                    Err(e) => return e,
                }
            }
            Stmt::Expr(expr, _semi) => {
                seq_elems.push(quote! { #expr });
            }
            Stmt::Macro(mac_stmt) => {
                let mac = &mac_stmt.mac;
                seq_elems.push(quote! { #mac });
            }
            Stmt::Item(_) => {
                // Rust item definitions inside staged_block! are unusual; pass
                // through but don't add to the staged sequence.
                let_decls.push(quote! { #stmt });
            }
        }
    }

    build(let_decls, seq_elems)
}

fn extract_ident(pat: &Pat) -> Result<proc_macro2::Ident, TokenStream2> {
    match pat {
        Pat::Ident(p) => Ok(p.ident.clone()),
        Pat::Type(p) => extract_ident(&p.pat),
        other => Err(syn::Error::new_spanned(
            other,
            "staged_block! let bindings must use a simple identifier pattern",
        )
        .to_compile_error()),
    }
}

fn build(let_decls: Vec<TokenStream2>, seq_elems: Vec<TokenStream2>) -> TokenStream2 {
    match seq_elems.len() {
        0 => quote! { { #(#let_decls)* () } },
        1 => {
            let elem = &seq_elems[0];
            quote! { { #(#let_decls)* #elem } }
        }
        _ => {
            quote! { { #(#let_decls)* (#(#seq_elems),*) } }
        }
    }
}
