//! Project Euler problems implemented as integration tests for the JIT compiler.
//!
//! Each test compiles a rust-lms function and verifies it against the published
//! answer for the official problem inputs, plus a smaller spot-check where useful.
//!
//! Tests marked `#[ignore]` document features missing from rust-lms — the body is
//! a stub explaining what's required.
//!
//! Conventions:
//!   * `Compiler::compile` consumes `self`, so each compiled function needs a
//!     fresh `Compiler::new()`.
//!   * `Compiled::as_fn` borrows the `Compiled`, so we always bind `compiled`
//!     before extracting `as_fn()`.
//!   * Slice indices/lengths are u64 (`SRefOps` requires `IntoStaged<U64Type>`);
//!     loop counters that touch a slice are u64 too. Pure arithmetic uses i64.
//!     There is no integer cast op yet, so the two worlds don't mix in a single
//!     expression.

use rust_lms::prelude::*;

// =============================================================================
// Euler 1: Sum of multiples of 3 or 5 below n.
// =============================================================================

#[test]
fn euler_01_multiples_of_3_or_5() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e01", |ctx, n: Var<i64>| {
        let i = ctx.var(1i64);
        let acc = ctx.var(0i64);
        ctx.while_loop(lt(i, n), move |ctx| {
            ctx.if_then(eq(rem(i, 3i64), 0i64), move |ctx| {
                ctx.store(acc, add(acc, i));
            });
            // Add i if divisible by 5 AND not divisible by 3 (already counted).
            ctx.if_then(eq(rem(i, 5i64), 0i64), move |ctx| {
                ctx.if_then(gt(rem(i, 3i64), 0i64), move |ctx| {
                    ctx.store(acc, add(acc, i));
                });
            });
            ctx.store(i, add(i, 1i64));
        });
        acc
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(10), 23);
    assert_eq!(g(1000), 233168); // official answer
}

// =============================================================================
// Euler 2: Sum of even Fibonacci terms ≤ limit.
// =============================================================================

#[test]
fn euler_02_even_fibonacci() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e02", |ctx, limit: Var<i64>| {
        let a = ctx.var(1i64);
        let b = ctx.var(2i64);
        let acc = ctx.var(0i64);
        let tmp = ctx.var(0i64);
        ctx.while_loop(lt(a, add(limit, 1i64)), move |ctx| {
            ctx.if_then(eq(rem(a, 2i64), 0i64), move |ctx| {
                ctx.store(acc, add(acc, a));
            });
            ctx.store(tmp, add(a, b));
            ctx.store(a, b);
            ctx.store(b, tmp);
        });
        acc
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(4_000_000), 4613732); // official answer
}

// =============================================================================
// Euler 3: Largest prime factor of n.
// =============================================================================

#[test]
fn euler_03_largest_prime_factor() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e03", |ctx, n: Var<i64>| {
        let m = ctx.var(0i64);
        ctx.store(m, n);
        let p = ctx.var(2i64);
        let largest = ctx.var(1i64);
        ctx.while_loop(lt(mul(p, p), add(m, 1i64)), move |ctx| {
            ctx.while_loop(eq(rem(m, p), 0i64), move |ctx| {
                ctx.store(largest, p);
                ctx.store(m, div(m, p));
            });
            ctx.store(p, add(p, 1i64));
        });
        ctx.if_then(gt(m, 1i64), move |ctx| {
            ctx.store(largest, m);
        });
        largest
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(13195), 29);
    assert_eq!(g(600_851_475_143), 6857); // official answer
}

// =============================================================================
// Euler 4: Largest palindrome product of two 3-digit numbers.
// =============================================================================

#[test]
fn euler_04_largest_palindrome_product() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("e04", |ctx| {
        let a = ctx.var(100i64);
        let best = ctx.var(0i64);
        ctx.while_loop(lt(a, 1000i64), move |ctx| {
            let b = ctx.var(100i64);
            ctx.while_loop(lt(b, 1000i64), move |ctx| {
                let n = ctx.var(0i64);
                ctx.store(n, mul(a, b));
                let r = ctx.var(0i64);
                let m = ctx.var(0i64);
                ctx.store(m, n);
                ctx.while_loop(gt(m, 0i64), move |ctx| {
                    ctx.store(r, add(mul(r, 10i64), rem(m, 10i64)));
                    ctx.store(m, div(m, 10i64));
                });
                ctx.if_then(eq(r, n), move |ctx| {
                    ctx.if_then(gt(n, best), move |ctx| {
                        ctx.store(best, n);
                    });
                });
                ctx.store(b, add(b, 1i64));
            });
            ctx.store(a, add(a, 1i64));
        });
        best
    });

    let compiled = compiler.compile(call0(f)).expect("compile");
    assert_eq!(compiled.run(), 906609); // official answer
}

// =============================================================================
// Euler 5: Smallest multiple of 1..n (LCM accumulated via Euclidean GCD).
// =============================================================================

#[test]
fn euler_05_smallest_multiple() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e05", |ctx, n: Var<i64>| {
        let acc = ctx.var(1i64);
        let i = ctx.var(2i64);
        ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
            let x = ctx.var(0i64);
            let y = ctx.var(0i64);
            let t = ctx.var(0i64);
            ctx.store(x, acc);
            ctx.store(y, i);
            ctx.while_loop(gt(y, 0i64), move |ctx| {
                ctx.store(t, rem(x, y));
                ctx.store(x, y);
                ctx.store(y, t);
            });
            ctx.store(acc, mul(div(acc, x), i));
            ctx.store(i, add(i, 1i64));
        });
        acc
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(10), 2520);
    assert_eq!(g(20), 232792560); // official answer
}

// =============================================================================
// Euler 6: Sum-square difference for 1..n.
// =============================================================================

#[test]
fn euler_06_sum_square_difference() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e06", |ctx, n: Var<i64>| {
        let i = ctx.var(1i64);
        let sum = ctx.var(0i64);
        let sq_sum = ctx.var(0i64);
        ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
            ctx.store(sum, add(sum, i));
            ctx.store(sq_sum, add(sq_sum, mul(i, i)));
            ctx.store(i, add(i, 1i64));
        });
        sub(mul(sum, sum), sq_sum)
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(10), 2640);
    assert_eq!(g(100), 25164150); // official answer
}

// =============================================================================
// Euler 7: Nth prime (1-indexed) via trial-division stream.
// =============================================================================

#[test]
fn euler_07_nth_prime() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e07", |ctx, n: Var<i64>| {
        let count = ctx.var(0i64);
        let candidate = ctx.var(1i64);
        let last_prime = ctx.var(0i64);
        ctx.while_loop(lt(count, n), move |ctx| {
            ctx.store(candidate, add(candidate, 1i64));
            let prime = ctx.var(1i64);
            let d = ctx.var(2i64);
            ctx.while_loop(lt(mul(d, d), add(candidate, 1i64)), move |ctx| {
                ctx.if_then(eq(rem(candidate, d), 0i64), move |ctx| {
                    ctx.store(prime, 0i64);
                });
                ctx.store(d, add(d, 1i64));
            });
            ctx.if_then(eq(prime, 1i64), move |ctx| {
                ctx.store(count, add(count, 1i64));
                ctx.store(last_prime, candidate);
            });
        });
        last_prime
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(6), 13);
    assert_eq!(g(10001), 104743); // official answer
}

// =============================================================================
// Euler 8: Largest product of K adjacent digits in a slice of digits.
// =============================================================================

#[test]
fn euler_08_largest_product_of_k_adjacent() {
    let mut compiler = Compiler::new();

    let f = compiler.fun2(
        "e08",
        |ctx, digits: Var<SRef<Slice<i64>>>, k: Var<U64Type>| {
            let n = ctx.var(0u64);
            ctx.store(n, digits.len());
            let i = ctx.var(0u64);
            let best = ctx.var(0i64);
            ctx.while_loop(lt(add(i, k), add(n, 1u64)), move |ctx| {
                let prod = ctx.var(1i64);
                let j = ctx.var(0u64);
                ctx.while_loop(lt(j, k), move |ctx| {
                    ctx.store(prod, mul(prod, digits.get_unchecked(add(i, j))));
                    ctx.store(j, add(j, 1u64));
                });
                ctx.if_then(gt(prod, best), move |ctx| {
                    ctx.store(best, prod);
                });
                ctx.store(i, add(i, 1u64));
            });
            best
        },
    );

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    let digits: Vec<i64> = include_str!("euler_08_data.txt")
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| (c as i64) - ('0' as i64))
        .collect();
    assert_eq!(digits.len(), 1000, "input file should hold 1000 digits");

    assert_eq!(g(&digits[..], 13u64), 23514624000); // official answer
    assert_eq!(g(&digits[..], 4u64), 5832); // published smaller case
}

// =============================================================================
// Euler 9: Pythagorean triplet with a+b+c=n.
// =============================================================================

#[test]
fn euler_09_pythagorean_triplet() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e09", |ctx, n: Var<i64>| {
        let result = ctx.var(0i64);
        let a = ctx.var(1i64);
        ctx.while_loop(lt(a, n), move |ctx| {
            let b = ctx.var(0i64);
            ctx.store(b, add(a, 1i64));
            ctx.while_loop(lt(b, n), move |ctx| {
                let c = ctx.var(0i64);
                ctx.store(c, sub(sub(n, a), b));
                ctx.if_then(gt(c, b), move |ctx| {
                    ctx.if_then(eq(add(mul(a, a), mul(b, b)), mul(c, c)), move |ctx| {
                        ctx.store(result, mul(mul(a, b), c));
                    });
                });
                ctx.store(b, add(b, 1i64));
            });
            ctx.store(a, add(a, 1i64));
        });
        result
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(1000), 31875000); // official answer
}

// =============================================================================
// Euler 10: Sum of primes below n.
// =============================================================================

#[test]
fn euler_10_sum_of_primes_below() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e10", |ctx, n: Var<i64>| {
        let sum = ctx.var(0i64);
        let i = ctx.var(2i64);
        ctx.while_loop(lt(i, n), move |ctx| {
            let prime = ctx.var(1i64);
            let d = ctx.var(2i64);
            ctx.while_loop(lt(mul(d, d), add(i, 1i64)), move |ctx| {
                ctx.if_then(eq(rem(i, d), 0i64), move |ctx| {
                    ctx.store(prime, 0i64);
                });
                ctx.store(d, add(d, 1i64));
            });
            ctx.if_then(eq(prime, 1i64), move |ctx| {
                ctx.store(sum, add(sum, i));
            });
            ctx.store(i, add(i, 1i64));
        });
        sum
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(10), 17);
    assert_eq!(g(2_000_000), 142913828922); // official answer
}

// =============================================================================
// Euler 12: First triangle number with more than t divisors.
// =============================================================================

#[test]
fn euler_12_highly_divisible_triangle() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e12", |ctx, t: Var<i64>| {
        let i = ctx.var(1i64);
        let tri = ctx.var(0i64);
        let result = ctx.var(0i64);
        ctx.while_loop(eq(result, 0i64), move |ctx| {
            ctx.store(tri, add(tri, i));
            let count = ctx.var(0i64);
            let d = ctx.var(1i64);
            ctx.while_loop(lt(mul(d, d), tri), move |ctx| {
                ctx.if_then(eq(rem(tri, d), 0i64), move |ctx| {
                    ctx.store(count, add(count, 2i64));
                });
                ctx.store(d, add(d, 1i64));
            });
            ctx.if_then(eq(mul(d, d), tri), move |ctx| {
                ctx.store(count, add(count, 1i64));
            });
            ctx.if_then(gt(count, t), move |ctx| {
                ctx.store(result, tri);
            });
            ctx.store(i, add(i, 1i64));
        });
        result
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(5), 28);
    assert_eq!(g(500), 76576500); // official answer
}

// =============================================================================
// Euler 14: Longest Collatz chain for n in 1..limit.
// =============================================================================

#[test]
fn euler_14_longest_collatz() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e14", |ctx, limit: Var<i64>| {
        let i = ctx.var(1i64);
        let best_len = ctx.var(0i64);
        let best_n = ctx.var(0i64);
        ctx.while_loop(lt(i, limit), move |ctx| {
            let x = ctx.var(0i64);
            ctx.store(x, i);
            let len = ctx.var(1i64);
            ctx.while_loop(gt(x, 1i64), move |ctx| {
                ctx.store(
                    x,
                    if_then_else(
                        eq(rem(x, 2i64), 0i64),
                        div(x, 2i64),
                        add(mul(x, 3i64), 1i64),
                    ),
                );
                ctx.store(len, add(len, 1i64));
            });
            ctx.if_then(gt(len, best_len), move |ctx| {
                ctx.store(best_len, len);
                ctx.store(best_n, i);
            });
            ctx.store(i, add(i, 1i64));
        });
        best_n
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(1_000_000), 837799); // official answer
}

// =============================================================================
// Euler 15: Lattice paths in an n×n grid = C(2n, n).
// =============================================================================

#[test]
fn euler_15_lattice_paths() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e15", |ctx, n: Var<i64>| {
        let i = ctx.var(1i64);
        let acc = ctx.var(1i64);
        ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
            ctx.store(acc, div(mul(acc, add(n, i)), i));
            ctx.store(i, add(i, 1i64));
        });
        acc
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(2), 6);
    assert_eq!(g(20), 137846528820); // official answer
}

// =============================================================================
// Euler 21: Sum of amicable numbers below n.
//
// Two-pass JIT: pass 1 fills sigma[i] = sum of proper divisors of i into a
// caller-allocated mut slice; pass 2 sums pairs where sigma(sigma(a)) = a.
// All values fit in u64; two `Compiler` instances since compile() consumes self.
// =============================================================================

#[test]
fn euler_21_amicable_sum() {
    let fill = {
        let mut compiler = Compiler::new();
        let fill_sigma = compiler.fun1("e21_sigma", |ctx, sigma: Var<SRefMut<Slice<U64Type>>>| {
            let n = ctx.var(0u64);
            ctx.store(n, sigma.len());
            let i = ctx.var(2u64);
            ctx.while_loop(lt(i, n), move |ctx| {
                let s = ctx.var(1u64);
                let d = ctx.var(2u64);
                ctx.while_loop(lt(mul(d, d), add(i, 1u64)), move |ctx| {
                    ctx.if_then(eq(rem(i, d), 0u64), move |ctx| {
                        ctx.store(s, add(s, d));
                        let q = ctx.var(0u64);
                        ctx.store(q, div(i, d));
                        ctx.if_then(gt(q, d), move |ctx| {
                            ctx.store(s, add(s, q));
                        });
                    });
                    ctx.store(d, add(d, 1u64));
                });
                ctx.emit(sigma.set_unchecked(i, s));
                ctx.store(i, add(i, 1u64));
            });
            Const::<UnitType>::new(())
        });
        compiler.compile(fill_sigma).expect("compile fill")
    };

    let sum = {
        let mut compiler = Compiler::new();
        let sum_amicable = compiler.fun1("e21_sum", |ctx, sigma: Var<SRef<Slice<U64Type>>>| {
            let n = ctx.var(0u64);
            ctx.store(n, sigma.len());
            let total = ctx.var(0u64);
            let a = ctx.var(2u64);
            ctx.while_loop(lt(a, n), move |ctx| {
                let b = ctx.var(0u64);
                ctx.store(b, sigma.get_unchecked(a));
                ctx.if_then(gt(b, a), move |ctx| {
                    ctx.if_then(lt(b, n), move |ctx| {
                        ctx.if_then(eq(sigma.get_unchecked(b), a), move |ctx| {
                            ctx.store(total, add(total, add(a, b)));
                        });
                    });
                });
                ctx.store(a, add(a, 1u64));
            });
            total
        });
        compiler.compile(sum_amicable).expect("compile sum")
    };

    let fill_fn = fill.as_fn();
    let sum_fn = sum.as_fn();

    let n: usize = 10_000;
    let mut sigma = vec![0u64; n];
    fill_fn(&mut sigma[..]);
    assert_eq!(sum_fn(&sigma[..]), 31626); // official answer
}

// =============================================================================
// Euler 25: First Fibonacci index with N digits.
//
// MISSING: big-integer support. Fib(4782) has 1000 digits and overflows i64
// long before that. Would need a `BigInt` staged type (Vec<u64> limbs) plus
// add/cmp/digit-count operations on it.
// =============================================================================

#[test]
#[ignore = "MISSING: big-integer type (no Vec<u64> limbs, no bigint add)"]
fn euler_25_thousand_digit_fibonacci() {
    unreachable!("ignored — see comment above");
}

// =============================================================================
// Euler 28: Sum of diagonals in a size×size number spiral.
// =============================================================================

#[test]
fn euler_28_spiral_diagonals() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e28", |ctx, size: Var<i64>| {
        let total = ctx.var(1i64);
        let k = ctx.var(3i64);
        ctx.while_loop(lt(k, add(size, 1i64)), move |ctx| {
            ctx.store(
                total,
                add(total, sub(mul(mul(k, k), 4i64), mul(sub(k, 1i64), 6i64))),
            );
            ctx.store(k, add(k, 2i64));
        });
        total
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(5), 101);
    assert_eq!(g(1001), 669171001); // official answer
}

// =============================================================================
// Euler 30: Sum of numbers equal to the sum of their digits raised to the 5th.
// =============================================================================

#[test]
fn euler_30_digit_fifth_powers() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("e30", |ctx, upper: Var<i64>| {
        let total = ctx.var(0i64);
        let n = ctx.var(2i64);
        ctx.while_loop(lt(n, upper), move |ctx| {
            let m = ctx.var(0i64);
            ctx.store(m, n);
            let sum = ctx.var(0i64);
            ctx.while_loop(gt(m, 0i64), move |ctx| {
                let d = ctx.var(0i64);
                ctx.store(d, rem(m, 10i64));
                let d5 = ctx.var(0i64);
                ctx.store(d5, mul(mul(mul(mul(d, d), d), d), d));
                ctx.store(sum, add(sum, d5));
                ctx.store(m, div(m, 10i64));
            });
            ctx.if_then(eq(sum, n), move |ctx| {
                ctx.store(total, add(total, n));
            });
            ctx.store(n, add(n, 1i64));
        });
        total
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    // 6 * 9^5 = 354294 is a safe upper bound.
    assert_eq!(g(354295), 443839); // official answer
}

// =============================================================================
// Euler 34: Sum of numbers equal to the sum of factorials of their digits.
//
// All loop counters are u64 so the digit `rem(m, 10u64)` can index `fact[]`
// directly. (No int-cast op yet to convert i64↔u64.)
// =============================================================================

#[test]
fn euler_34_digit_factorials() {
    let mut compiler = Compiler::new();

    let f = compiler.fun2(
        "e34",
        |ctx, upper: Var<U64Type>, fact: Var<SRef<Slice<U64Type>>>| {
            let total = ctx.var(0u64);
            let n = ctx.var(3u64);
            ctx.while_loop(lt(n, upper), move |ctx| {
                let m = ctx.var(0u64);
                ctx.store(m, n);
                let sum = ctx.var(0u64);
                ctx.while_loop(gt(m, 0u64), move |ctx| {
                    ctx.store(sum, add(sum, fact.get_unchecked(rem(m, 10u64))));
                    ctx.store(m, div(m, 10u64));
                });
                ctx.if_then(eq(sum, n), move |ctx| {
                    ctx.store(total, add(total, n));
                });
                ctx.store(n, add(n, 1u64));
            });
            total
        },
    );

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    let fact: [u64; 10] = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880];
    // Safe upper bound: 7 * 9! = 2540160.
    assert_eq!(g(2540160u64, &fact[..]), 40730); // official answer
}

// =============================================================================
// Euler 48: Last 10 digits of 1^1 + 2^2 + ... + 1000^1000.
//
// MISSING: 64-bit modular multiplication. (a*b) mod 10^10 requires either
// u128/i128 arithmetic or a dedicated 128-bit mul helper; a single i64
// multiply of two ~2^34 operands overflows i64.
// =============================================================================

#[test]
#[ignore = "MISSING: 128-bit multiply or modmul helper for 10^10 modulus"]
fn euler_48_self_powers() {
    unreachable!("ignored — see comment above");
}

// =============================================================================
// Euler 67: Max path sum in a triangle (bottom-up DP into a mut workspace).
//
// Layout: `tri` is row-major flat (row r starts at offset r*(r+1)/2, width r+1).
// `workspace` is a caller-allocated [i64; num_rows] used for the DP frontier.
// Loop counters are u64 (slice indexing); DP values are i64.
// =============================================================================

#[test]
fn euler_67_max_path_sum_triangle() {
    let mut compiler = Compiler::new();

    let f = compiler.fun3(
        "e67",
        |ctx,
         tri: Var<SRef<Slice<i64>>>,
         workspace: Var<SRefMut<Slice<i64>>>,
         num_rows: Var<U64Type>| {
            // Seed workspace with the bottom row.
            let last_row_offset = ctx.var(0u64);
            ctx.store(
                last_row_offset,
                div(mul(sub(num_rows, 1u64), num_rows), 2u64),
            );
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, num_rows), move |ctx| {
                ctx.emit(workspace.set_unchecked(i, tri.get_unchecked(add(last_row_offset, i))));
                ctx.store(i, add(i, 1u64));
            });

            // Fold from row num_rows-2 down to row 0. We track row+1 to keep the
            // counter unsigned (it never reaches 0 during the loop body).
            let row_plus_1 = ctx.var(0u64);
            ctx.store(row_plus_1, sub(num_rows, 1u64));
            ctx.while_loop(gt(row_plus_1, 0u64), move |ctx| {
                let row = ctx.var(0u64);
                ctx.store(row, sub(row_plus_1, 1u64));
                let row_offset = ctx.var(0u64);
                ctx.store(row_offset, div(mul(row, add(row, 1u64)), 2u64));
                let j = ctx.var(0u64);
                ctx.while_loop(lt(j, add(row, 1u64)), move |ctx| {
                    let l = ctx.var(0i64);
                    let r = ctx.var(0i64);
                    ctx.store(l, workspace.get_unchecked(j));
                    ctx.store(r, workspace.get_unchecked(add(j, 1u64)));
                    let best = ctx.var(0i64);
                    ctx.store(best, select(gt(l, r), l, r));
                    ctx.emit(
                        workspace
                            .set_unchecked(j, add(tri.get_unchecked(add(row_offset, j)), best)),
                    );
                    ctx.store(j, add(j, 1u64));
                });
                ctx.store(row_plus_1, sub(row_plus_1, 1u64));
            });
            workspace.get_unchecked(0u64)
        },
    );

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    // Euler-18-style small triangle.
    //         3
    //       7  4
    //     2  4  6
    //   8  5  9  3
    let small: [i64; 10] = [3, 7, 4, 2, 4, 6, 8, 5, 9, 3];
    let mut ws = vec![0i64; 4];
    assert_eq!(g(&small[..], &mut ws[..], 4u64), 23);

    // Official Euler-18 15-row triangle (answer = 1074).
    let e18_rows: [&[i64]; 15] = [
        &[75],
        &[95, 64],
        &[17, 47, 82],
        &[18, 35, 87, 10],
        &[20, 4, 82, 47, 65],
        &[19, 1, 23, 75, 3, 34],
        &[88, 2, 77, 73, 7, 63, 67],
        &[99, 65, 4, 28, 6, 16, 70, 92],
        &[41, 41, 26, 56, 83, 40, 80, 70, 33],
        &[41, 48, 72, 33, 47, 32, 37, 16, 94, 29],
        &[53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14],
        &[70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57],
        &[91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48],
        &[63, 66, 4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31],
        &[4, 62, 98, 27, 23, 9, 70, 98, 73, 93, 38, 53, 60, 4, 23],
    ];
    let flat: Vec<i64> = e18_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let mut ws = vec![0i64; 15];
    assert_eq!(g(&flat[..], &mut ws[..], 15u64), 1074);
}
