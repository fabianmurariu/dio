mod common;
use common::{for_each_backend, with_backends};
use rust_lms::prelude::*;

#[test]
fn test_c_option_layout() {
    // Verify layout matches our assumptions
    assert_eq!(std::mem::size_of::<COption<i64>>(), 16);
    assert_eq!(std::mem::align_of::<COption<i64>>(), 8);

    // Verify discriminant values
    let none: COption<i64> = COption::None;
    let some: COption<i64> = COption::Some(42);

    // Check that discriminant is at offset 0
    let none_ptr = &none as *const COption<i64> as *const u64;
    let some_ptr = &some as *const COption<i64> as *const u64;

    unsafe {
        assert_eq!(*none_ptr, 0); // None discriminant
        assert_eq!(*some_ptr, 1); // Some discriminant
    }
}

#[test]
fn test_c_some_i64() {
    for_each_backend(|compiler| {
        // Create COption::Some(42)
        let expr = c_some::<i64, _>(42i64);
        let wrapped = unwrap_or(expr, 0i64);

        let compiled = compiler.compile(wrapped).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    });
}

#[test]
fn test_c_none_i64() {
    for_each_backend(|compiler| {
        // Create COption::None, unwrap_or should return default
        let expr = c_none::<i64>();
        let wrapped = unwrap_or(expr, 99i64);

        let compiled = compiler.compile(wrapped).expect("compilation failed");
        assert_eq!(compiled.run(), 99);
    });
}

#[test]
fn test_is_some() {
    for_each_backend(|compiler| {
        let some_expr = c_some::<i64, _>(42i64);
        let check = is_some(some_expr);

        let compiled = compiler.compile(check).expect("compilation failed");
        assert!(compiled.run());
    });
}

#[test]
fn test_is_none() {
    for_each_backend(|compiler| {
        let none_expr = c_none::<i64>();
        let check = is_none(none_expr);

        let compiled = compiler.compile(check).expect("compilation failed");
        assert!(compiled.run());
    });
}

#[test]
fn test_match_opt_some() {
    for_each_backend(|mut compiler| {
        // match Some(10) { Some(x) => x + 5, None => 0 }
        let func = compiler.fun1("test", |ctx, _dummy: Var<i64>| {
            let opt = c_some::<i64, _>(10i64);
            match_opt(ctx, opt, |_ctx, val| add(val, 5i64), Const::<i64>::new(0))
        });

        let compiled = compiler
            .compile(call1(func, 0i64))
            .expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    });
}

#[test]
fn test_match_opt_none() {
    for_each_backend(|mut compiler| {
        // match None { Some(x) => x + 5, None => 99 }
        let func = compiler.fun1("test", |ctx, _dummy: Var<i64>| {
            let opt = c_none::<i64>();
            match_opt(ctx, opt, |_ctx, val| add(val, 5i64), Const::<i64>::new(99))
        });

        let compiled = compiler
            .compile(call1(func, 0i64))
            .expect("compilation failed");
        assert_eq!(compiled.run(), 99);
    });
}

#[test]
fn test_coption_from_option() {
    let some: COption<i64> = Some(42).into();
    assert_eq!(some, COption::Some(42));

    let none: COption<i64> = None.into();
    assert_eq!(none, COption::None);
}

#[test]
fn test_option_from_coption() {
    let some: Option<i64> = COption::Some(42).into();
    assert_eq!(some, Some(42));

    let none: Option<i64> = COption::<i64>::None.into();
    assert_eq!(none, None);
}

// =========================================================================
// Function Pointer Tests: COption<i64>
// =========================================================================

#[test]
fn test_fn_taking_coption_i64() {
    for_each_backend(|mut compiler| {
        // fn unwrap_or_default(opt: COption<i64>) -> i64
        let unwrap_fn = compiler.fun1("unwrap_or_default", |_ctx, opt: Var<COptionType<i64>>| {
            unwrap_or(opt, -1i64)
        });

        let compiled = compiler.compile(unwrap_fn).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with Some
        assert_eq!(f.call(COption::Some(42)), 42);
        assert_eq!(f.call(COption::Some(0)), 0);
        assert_eq!(f.call(COption::Some(-100)), -100);

        // Test with None
        assert_eq!(f.call(COption::None), -1);
    });
}

#[test]
fn test_fn_returning_coption_i64() {
    for_each_backend(|mut compiler| {
        // fn maybe_double(x: i64) -> COption<i64>
        // Returns Some(x * 2) if x > 0, else None
        let maybe_double = compiler.fun1("maybe_double", |ctx, x: Var<i64>| {
            let doubled = c_some::<i64, _>(mul(x, 2i64));
            let none = c_none::<i64>();
            // if x > 0 then Some(x*2) else None
            match_opt(
                ctx,
                if_then_else(lt(0i64, x), doubled, none),
                |_ctx, val| c_some::<i64, _>(val),
                c_none::<i64>(),
            )
        });

        let compiled = compiler.compile(maybe_double).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(5), COption::Some(10));
        assert_eq!(f.call(1), COption::Some(2));
        assert_eq!(f.call(0), COption::None);
        assert_eq!(f.call(-5), COption::None);
    });
}

#[test]
fn test_fn_coption_i64_roundtrip() {
    for_each_backend(|mut compiler| {
        // fn add_one_if_some(opt: COption<i64>) -> COption<i64>
        let add_one = compiler.fun1("add_one_if_some", |ctx, opt: Var<COptionType<i64>>| {
            match_opt(
                ctx,
                opt,
                |_ctx, val| c_some::<i64, _>(add(val, 1i64)),
                c_none::<i64>(),
            )
        });

        let compiled = compiler.compile(add_one).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(10)), COption::Some(11));
        assert_eq!(f.call(COption::Some(-1)), COption::Some(0));
        assert_eq!(f.call(COption::None), COption::None);
    });
}

// =========================================================================
// Function Pointer Tests: COption<f64>
// =========================================================================

#[test]
fn test_c_option_f64_layout() {
    // Verify layout for f64 variant
    assert_eq!(std::mem::size_of::<COption<f64>>(), 16);
    assert_eq!(std::mem::align_of::<COption<f64>>(), 8);

    let some: COption<f64> = COption::Some(3.15);
    let ptr = &some as *const COption<f64> as *const u8;
    unsafe {
        // Discriminant at offset 0
        let disc = *(ptr as *const u64);
        assert_eq!(disc, 1, "discriminant should be 1 for Some");
        // Value at offset 8
        let val = *((ptr.add(8)) as *const f64);
        assert_eq!(val, 3.15, "value should be 3.15");
    }
}

#[test]
fn test_fn_taking_coption_f64() {
    for_each_backend(|mut compiler| {
        // fn unwrap_or_zero(opt: COption<f64>) -> f64
        let unwrap_fn = compiler.fun1("unwrap_or_zero", |_ctx, opt: Var<COptionType<f64>>| {
            unwrap_or(opt, 0.0f64)
        });

        let compiled = compiler.compile(unwrap_fn).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(3.15)), 3.15);
        assert_eq!(f.call(COption::Some(-2.5)), -2.5);
        assert_eq!(f.call(COption::None), 0.0);
    });
}

#[test]
fn test_fn_returning_coption_f64() {
    for_each_backend(|mut compiler| {
        // fn wrap_f64(x: f64) -> COption<f64>
        // Always returns Some(x)
        let wrap = compiler.fun1("wrap_f64", |_ctx, x: Var<f64>| c_some::<f64, _>(x));

        let compiled = compiler.compile(wrap).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(3.15), COption::Some(3.15));
        assert_eq!(f.call(0.0), COption::Some(0.0));
        assert_eq!(f.call(-1.5), COption::Some(-1.5));
    });
}

// =========================================================================
// Function Pointer Tests: OptRefType (Option<&T>)
// =========================================================================

#[test]
fn test_fn_taking_opt_ref_i64() {
    for_each_backend(|mut compiler| {
        // fn deref_or_default(opt: Option<&i64>) -> i64
        let deref_fn = compiler.fun1("deref_or_default", |ctx, opt: Var<OptRefType<i64>>| {
            // if some, load the value; else return -1
            match_opt_ref(ctx, opt, |_ctx, ptr| load_ref(ptr), Const::<i64>::new(-1))
        });

        let compiled = compiler.compile(deref_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let val = 42i64;
        assert_eq!(f.call(Some(&val)), 42);

        let val2 = -100i64;
        assert_eq!(f.call(Some(&val2)), -100);

        assert_eq!(f.call(None), -1);
    });
}

#[test]
fn test_fn_returning_opt_ref_i64() {
    for_each_backend(|mut compiler| {
        // fn make_ref(ptr: &i64) -> Option<&i64>
        // Just wraps the reference in Some
        let make_ref = compiler.fun1("make_ref", |_ctx, ptr: Var<SRef<i64>>| {
            opt_ref_some::<i64, _>(ptr)
        });

        let compiled = compiler.compile(make_ref).expect("compilation failed");
        let f = compiled.as_fn();

        let val = 99i64;
        let result = f.call(&val);
        assert_eq!(result, Some(&99i64));
    });
}

#[test]
fn test_fn_opt_ref_conditional() {
    for_each_backend(|mut compiler| {
        // fn ref_if_positive(ptr: &i64) -> Option<&i64>
        // Returns Some(ptr) if *ptr > 0, else None
        let ref_if_pos = compiler.fun1("ref_if_positive", |_ctx, ptr: Var<SRef<i64>>| {
            let val = load_ref(ptr);
            if_then_else(
                lt(0i64, val), // val > 0
                opt_ref_some::<i64, _>(ptr),
                opt_ref_none::<i64>(),
            )
        });

        let compiled = compiler.compile(ref_if_pos).expect("compilation failed");
        let f = compiled.as_fn();

        let pos = 42i64;
        assert_eq!(f.call(&pos), Some(&42i64));

        let zero = 0i64;
        assert_eq!(f.call(&zero), None);

        let neg = -10i64;
        assert_eq!(f.call(&neg), None);
    });
}

// =========================================================================
// Function Pointer Tests: OptMutRefType (Option<&mut T>)
// =========================================================================

#[test]
fn test_fn_taking_opt_mut_ref_i64() {
    for_each_backend(|mut compiler| {
        // fn read_and_double(opt: Option<&mut i64>) -> i64
        // If Some, reads the value and returns it doubled (without mutating); else returns -1
        let read_fn = compiler.fun1("read_and_double", |ctx, opt: Var<OptMutRefType<i64>>| {
            match_opt_mut_ref(
                ctx,
                opt,
                |_ctx, mut ptr| mul(load_ref_mut(&mut ptr), 2i64),
                Const::<i64>::new(-1),
            )
        });

        let compiled = compiler.compile(read_fn).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with Some - just reading, not mutating
        let mut val = 21i64;
        assert_eq!(f.call(Some(&mut val)), 42);

        let mut val2 = 5i64;
        assert_eq!(f.call(Some(&mut val2)), 10);

        // None case
        assert_eq!(f.call(None), -1);
    });
}

#[test]
fn test_fn_mutating_opt_mut_ref_i64() {
    // Two kernels rather than one: a `match_opt_*` arm is a single staged
    // expression, so mutating *and* reporting the new value are separate.
    with_backends(|make| {
        // fn increment_in_place(opt: Option<&mut i64>) — Some increments in place.
        let mut c = make();
        let incr = c.fun1("increment_in_place", |ctx, opt: Var<OptMutRefType<i64>>| {
            match_opt_mut_ref(
                ctx,
                opt,
                |_ctx, mut ptr| {
                    // Sequential *Rust* bindings so the two reborrows of `ptr`
                    // do not overlap; the staged arm is still one expression.
                    let loaded = load_ref_mut(&mut ptr);
                    store_ref(&mut ptr, add(loaded, 1i64))
                },
                unit(),
            )
        });
        let compiled = c.compile(incr).expect("compilation failed");
        let f = compiled.as_fn();

        let mut val = 41i64;
        f.call(Some(&mut val));
        assert_eq!(val, 42, "Some mutates in place");

        f.call(None); // no target to mutate
        // fn incremented_or(opt: Option<&mut i64>) -> i64 — the value, or -1.
        let mut c = make();
        let read = c.fun1("incremented_or", |ctx, opt: Var<OptMutRefType<i64>>| {
            match_opt_mut_ref(
                ctx,
                opt,
                |_ctx, mut ptr| add(load_ref_mut(&mut ptr), 1i64),
                Const::<i64>::new(-1),
            )
        });
        let compiled = c.compile(read).expect("compilation failed");
        let f = compiled.as_fn();

        let mut v = 41i64;
        assert_eq!(f.call(Some(&mut v)), 42);
        assert_eq!(f.call(None), -1);
    });
}

#[test]
fn test_fn_returning_opt_mut_ref_i64() {
    for_each_backend(|mut compiler| {
        // fn make_mut_ref(ptr: &mut i64) -> Option<&mut i64>
        let make_ref = compiler.fun1("make_mut_ref", |_ctx, ptr: Var<SRefMut<i64>>| {
            opt_mut_ref_some::<i64, _>(ptr)
        });

        let compiled = compiler.compile(make_ref).expect("compilation failed");
        let f = compiled.as_fn();

        let mut val = 99i64;
        let result = f.call(&mut val);
        assert!(result.is_some());
        if let Some(r) = result {
            assert_eq!(*r, 99);
            *r = 100;
        }
        assert_eq!(val, 100);
    });
}

// =========================================================================
// Function Pointer Tests: COption<f64> advanced
// =========================================================================

#[test]
fn test_fn_coption_f64_roundtrip() {
    for_each_backend(|mut compiler| {
        // fn square_if_some(opt: COption<f64>) -> COption<f64>
        let square_fn = compiler.fun1("square_if_some", |ctx, opt: Var<COptionType<f64>>| {
            match_opt(
                ctx,
                opt,
                |_ctx, val| c_some::<f64, _>(mul(val, val)),
                c_none::<f64>(),
            )
        });

        let compiled = compiler.compile(square_fn).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(3.0)), COption::Some(9.0));
        assert_eq!(f.call(COption::Some(-2.0)), COption::Some(4.0));
        assert_eq!(f.call(COption::None), COption::None);
    });
}

// =========================================================================
// Function Pointer Tests: Multi-argument functions with options
// =========================================================================

#[test]
fn test_fn2_with_coption() {
    for_each_backend(|mut compiler| {
        // fn add_options(a: COption<i64>, b: COption<i64>) -> COption<i64>
        // Returns Some(a + b) if both are Some, else None
        let add_opts = compiler.fun2(
            "add_options",
            |ctx, a: Var<COptionType<i64>>, b: Var<COptionType<i64>>| {
                match_opt(
                    ctx,
                    a,
                    |ctx, a_val| {
                        match_opt(
                            ctx,
                            b,
                            |_ctx, b_val| c_some::<i64, _>(add(a_val, b_val)),
                            c_none::<i64>(),
                        )
                    },
                    c_none::<i64>(),
                )
            },
        );

        let compiled = compiler.compile(add_opts).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(
            f.call(COption::Some(10), COption::Some(20)),
            COption::Some(30)
        );
        assert_eq!(f.call(COption::Some(5), COption::None), COption::None);
        assert_eq!(f.call(COption::None, COption::Some(5)), COption::None);
        assert_eq!(f.call(COption::None, COption::None), COption::None);
    });
}

#[test]
fn test_fn2_mixed_option_and_primitive() {
    for_each_backend(|mut compiler| {
        // fn unwrap_or_add(opt: COption<i64>, default: i64) -> i64
        let unwrap_add = compiler.fun2(
            "unwrap_or_add",
            |_ctx, opt: Var<COptionType<i64>>, default: Var<i64>| unwrap_or(opt, default),
        );

        let compiled = compiler.compile(unwrap_add).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(42), 0), 42);
        assert_eq!(f.call(COption::None, 99), 99);
        assert_eq!(f.call(COption::Some(10), 99), 10);
    });
}
