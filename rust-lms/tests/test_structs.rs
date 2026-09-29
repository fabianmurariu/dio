//! Integration tests for struct support with derive macro
//!
//! These tests verify that structs are passed BY VALUE:
//! - `Var<Point>` means `fn(Point)` - pass struct by value
//! - `Var<SRef<Point>>` would mean `fn(&Point)` - pass by reference (not tested here)

use rust_lms::prelude::*;
use rust_lms::refer::{SRef, SRefMut};

mod common;
use common::for_each_backend;

// A borrowed struct: `#[derive(StagedType)]` on a struct with one lifetime
// generates the lifetime-free marker `FfiAdjListStaged` for kernels, and each
// call takes `FfiAdjList<'call>`, so short-lived host data can be passed.
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct FfiAdjList<'a> {
    #[staged(SRef<Slice<u64>>)]
    offsets: &'a [u64],
    #[staged(SRef<Slice<u64>>)]
    neighbours: &'a [u64],
    #[staged(u64)]
    num_nodes: u64,
}

#[test]
fn test_borrowed_struct_of_slices() {
    for_each_backend(|mut compiler| {
        // Sum of degrees: offsets[num_nodes] - offsets[0].
        let sum_deg = compiler.fun1("sum_degrees", |ctx, g: Var<FfiAdjListStaged>| {
            let offsets = g.get(FfiAdjListType::offsets());
            let n = g.get(FfiAdjListType::num_nodes());
            range(0, n).fold(ctx, 0, |acc, i| {
                let deg = unsafe { offsets.get_unchecked(i + 1) - offsets.get_unchecked(i) };
                acc + deg
            })
        });
        let compiled = compiler.compile(sum_deg).expect("compilation failed");

        // One kernel, fresh short-lived graphs on every call.
        for extra in 0..3u64 {
            let offsets = vec![0u64, 1, 3 + extra];
            let neighbours: Vec<u64> = (0..3 + extra).collect();
            let graph = FfiAdjList {
                offsets: &offsets,
                neighbours: &neighbours,
                num_nodes: 2,
            };
            assert_eq!(compiled.call(graph), neighbours.len() as u64);
        }
    });
}

#[test]
fn test_borrowed_struct_returned_by_value() {
    for_each_backend(|mut compiler| {
        let id = compiler.fun1("id", |_ctx, g: Var<FfiAdjListStaged>| g);
        let compiled = compiler.compile(id).expect("compilation failed");
        let offsets = vec![0u64, 2];
        let neighbours = vec![7u64, 8];
        let graph = FfiAdjList {
            offsets: &offsets,
            neighbours: &neighbours,
            num_nodes: 1,
        };
        let back = compiled.call(graph);
        assert_eq!(back.neighbours, &[7, 8]);
        assert_eq!(back.num_nodes, 1);
    });
}

/// Stage-0 view of an [`FfiAdjList`] inside a kernel: its fields, bound once.
///
/// Row `i` is the staged slice `neighbours[offsets[i]..offsets[i + 1]]`, and
/// [`rows`](Self::rows) iterates them with random access, so it composes with
/// `enumerate` and `zip` like a slice iterator does.
#[derive(Clone, Copy)]
struct AdjListStaged {
    offsets: Var<SRef<Slice<u64>>>,
    neighbours: Var<SRef<Slice<u64>>>,
    num_nodes: Var<u64>,
}

impl AdjListStaged {
    /// # Safety
    ///
    /// At execution the graph must be a valid CSR: `offsets` holds at least
    /// `num_nodes + 1` non-decreasing entries, and `offsets[num_nodes] <=
    /// neighbours.len()`. Rows are read without bounds checks.
    unsafe fn new(ctx: &mut Ctx, graph: Var<FfiAdjListStaged>) -> Self {
        AdjListStaged {
            offsets: ctx.bind(graph.get(FfiAdjListType::offsets())),
            neighbours: ctx.bind(graph.get(FfiAdjListType::neighbours())),
            num_nodes: ctx.bind(graph.get(FfiAdjListType::num_nodes())),
        }
    }

    fn num_nodes(self) -> Var<u64> {
        self.num_nodes
    }

    /// The adjacency list of `node`.
    ///
    /// # Safety
    ///
    /// `node < num_nodes` at execution.
    unsafe fn row(self, ctx: &mut Ctx, node: Var<u64>) -> Var<SRef<Slice<u64>>> {
        // SAFETY: `node < num_nodes` (caller) and the CSR invariant of `new`
        // make both offsets readable and `start <= end <= neighbours.len()`.
        unsafe {
            let start = ctx.bind(self.offsets.get_unchecked(node));
            let end = ctx.bind(self.offsets.get_unchecked(node + 1u64));
            ctx.bind(self.neighbours.subslice_unchecked(start, end))
        }
    }

    /// Iterate the rows, in node order.
    fn rows(self) -> AdjRows {
        AdjRows { graph: self }
    }
}

/// Iterator over the rows of an [`AdjListStaged`]; item `i` is row `i`.
#[derive(Clone, Copy)]
struct AdjRows {
    graph: AdjListStaged,
}

impl StagedIterator for AdjRows {
    type Item = SRef<Slice<u64>>;
    type Cursor = AdjRowsCursor;

    fn open(self, ctx: &mut Ctx) -> AdjRowsCursor {
        AdjRowsCursor {
            graph: self.graph,
            pos: ctx.var(0u64),
        }
    }
}

impl IndexedStagedIterator for AdjRows {
    type LenExpr = Var<u64>;

    fn len(&self) -> Var<u64> {
        self.graph.num_nodes()
    }
}

struct AdjRowsCursor {
    graph: AdjListStaged,
    pos: Var<u64>,
}

impl Cursor for AdjRowsCursor {
    type Item = SRef<Slice<u64>>;
    type Close = ();

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<SRef<Slice<u64>>>, ()) {
        let pos = self.pos;
        ctx.exit_if(ge(pos, self.graph.num_nodes()), done);
        // SAFETY: the exit above proves `pos < num_nodes`.
        let row = unsafe { self.graph.row(ctx, pos) };
        ctx.store(pos, pos + 1u64);
        (row, ())
    }

    fn indexed_len(&mut self, _ctx: &mut Ctx) -> Option<Var<u64>> {
        Some(self.graph.num_nodes())
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<SRef<Slice<u64>>>, ()) {
        // SAFETY: forwarded from the caller (`index < num_nodes`).
        (unsafe { self.graph.row(ctx, index) }, ())
    }
}

/// A 4-node CSR graph: 0 -> {1, 2}, 1 -> {}, 2 -> {0, 1, 3}, 3 -> {2}.
fn small_csr() -> (Vec<u64>, Vec<u64>) {
    (vec![0, 2, 2, 5, 6], vec![1, 2, 0, 1, 3, 2])
}

#[test]
fn test_adj_rows_enumerate_degrees() {
    for_each_backend(|mut compiler| {
        let degrees = compiler.fun2(
            "degrees",
            |ctx, graph: Var<FfiAdjListStaged>, out: Var<SRefMut<Slice<u64>>>| {
                // SAFETY: the host passes a valid CSR.
                let graph = unsafe { AdjListStaged::new(ctx, graph) };
                graph.rows().enumerate().for_each(ctx, |ctx, node, row| {
                    // SAFETY: `out` has one slot per node.
                    ctx.emit(unsafe { out.set_unchecked(node, row.len()) });
                });
                Const::<()>::new(())
            },
        );
        let compiled = compiler.compile(degrees).expect("compilation failed");

        let (offsets, neighbours) = small_csr();
        let mut out = vec![0u64; 4];
        let graph = FfiAdjList {
            offsets: &offsets,
            neighbours: &neighbours,
            num_nodes: 4,
        };
        compiled.call(graph, &mut out);
        assert_eq!(out, [2, 0, 3, 1]);
    });
}

#[test]
fn test_adj_rows_nested_iteration() {
    for_each_backend(|mut compiler| {
        // Sum over nodes of (node * sum of its neighbour ids).
        let weighted = compiler.fun1("weighted", |ctx, graph: Var<FfiAdjListStaged>| {
            // SAFETY: the host passes a valid CSR.
            let graph = unsafe { AdjListStaged::new(ctx, graph) };
            let total = ctx.var(0u64);
            graph.rows().enumerate().for_each(ctx, |ctx, node, row| {
                let row_sum = row.staged_iter().sum(ctx);
                ctx.store(total, total + node * row_sum);
            });
            total
        });
        let compiled = compiler.compile(weighted).expect("compilation failed");

        // One kernel over fresh graphs: the full graph, then an empty one.
        let (offsets, neighbours) = small_csr();
        let graph = FfiAdjList {
            offsets: &offsets,
            neighbours: &neighbours,
            num_nodes: 4,
        };
        // 0*(1+2) + 1*0 + 2*(0+1+3) + 3*2
        assert_eq!(compiled.call(graph), 14);

        let empty = vec![0u64];
        let graph = FfiAdjList {
            offsets: &empty,
            neighbours: &[],
            num_nodes: 0,
        };
        assert_eq!(compiled.call(graph), 0);
    });
}

#[test]
fn test_adj_rows_zip_random_access() {
    for_each_backend(|mut compiler| {
        // zip with a per-node weight slice drives the rows by index (`next_at`).
        let score = compiler.fun2(
            "score",
            |ctx, graph: Var<FfiAdjListStaged>, weights: Var<SRef<Slice<u64>>>| {
                // SAFETY: the host passes a valid CSR.
                let graph = unsafe { AdjListStaged::new(ctx, graph) };
                let total = ctx.var(0u64);
                graph.rows().zip(weights).for_each(ctx, |ctx, row, weight| {
                    ctx.store(total, total + row.len() * weight);
                });
                total
            },
        );
        let compiled = compiler.compile(score).expect("compilation failed");

        let (offsets, neighbours) = small_csr();
        let graph = FfiAdjList {
            offsets: &offsets,
            neighbours: &neighbours,
            num_nodes: 4,
        };
        // degrees [2, 0, 3, 1] . weights [1, 10, 100]: zip stops at 3 rows.
        assert_eq!(compiled.call(graph, &[1, 10, 100]), 2 + 300);
    });
}

// Test with simple Copy struct
// Note: Structs MUST be Copy for pass-by-value semantics
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Point {
    #[staged(i64)]
    x: i64,
    #[staged(f64)]
    y: f64,
}

#[test]
fn test_scoped_mutable_field_operations() {
    for_each_backend(|mut compiler| {
        let update = compiler.fun1("update_point", |ctx, mut point: Var<SRefMut<Point>>| {
            {
                let mut x = field_mut(&mut point, PointType::x());
                let old = ctx.bind(x.load());
                ctx.emit(x.store(old + 1i64));
            }

            load_field_mut(&mut point, PointType::x())
        });

        let compiled = compiler.compile(update).expect("compilation failed");
        let mut point = Point { x: 41, y: 3.15 };
        assert_eq!(compiled.call(&mut point), 42);
        assert_eq!(point.x, 42);
    });
}

#[test]
fn test_split_disjoint_mutable_fields() {
    for_each_backend(|mut compiler| {
        let update = compiler.fun1("split_point", |ctx, point: Var<SRefMut<Point>>| {
            let (x, y) = split_fields_mut(point, PointType::x(), PointType::y());
            ctx.emit(store_ref(x, Const::<i64>::new(17)));
            ctx.emit(store_ref(y, Const::<f64>::new(2.5)));
            Const::<()>::new(())
        });

        let compiled = compiler.compile(update).expect("compilation failed");
        let mut point = Point { x: 0, y: 0.0 };
        compiled.call(&mut point);
        assert_eq!(point.x, 17);
        assert_eq!(point.y, 2.5);
    });
}

#[test]
fn test_simple_struct_field_access() {
    for_each_backend(|mut compiler| {
        // Create a function that reads the x field from a Point
        // fn get_x(pt: Point) -> i64  -- NOTE: Pass by VALUE!
        let get_x = compiler.fun1("get_x", |_ctx, pt: Var<Point>| pt.get(PointType::x()));

        let compiled = compiler.compile(get_x).expect("compilation failed");
        let f = compiled.as_fn();

        // Create a test point - passed by VALUE
        let point = Point { x: 42, y: 3.15 };
        let result = f.call(point); // Pass by value, not &point

        assert_eq!(result, 42);
    });
}

#[test]
fn test_struct_multiple_fields() {
    for_each_backend(|mut compiler| {
        // fn sum_fields(pt: Point) -> i64
        let sum_fields = compiler.fun1("sum_fields", |_ctx, pt: Var<Point>| {
            let x = pt.get(PointType::x());
            let _y = pt.get(PointType::y()); // Just to verify we can access y

            // Add 3 to x for testing
            x + 3i64
        });

        let compiled = compiler.compile(sum_fields).expect("compilation failed");
        let f = compiled.as_fn();

        let point = Point { x: 10, y: 3.15 };
        let result = f.call(point); // Pass by value

        assert_eq!(result, 13); // 10 + 3
    });
}

#[test]
fn test_struct_pass_by_value_semantics() {
    for_each_backend(|mut compiler| {
        // This test verifies that structs are truly passed by value
        // fn read_x(pt: Point) -> i64
        let read_x = compiler.fun1("read_x", |_ctx, pt: Var<Point>| pt.get(PointType::x()));

        let compiled = compiler.compile(read_x).expect("compilation failed");
        let f = compiled.as_fn();

        let point = Point { x: 99, y: 2.71 };
        let result = f.call(point); // Pass by value

        assert_eq!(result, 99);
    });
}

// Test nested struct (struct with struct field)
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Inner {
    #[staged(i64)]
    value: i64,
}

#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Outer {
    #[staged(Inner)]
    inner: Inner,
    #[staged(i64)]
    extra: i64,
}

#[test]
fn test_nested_struct_access() {
    for_each_backend(|mut compiler| {
        // fn get_inner_value(outer: Outer) -> i64
        // Access: outer.inner.value
        // Note: Outer is passed by VALUE, so we use .field() (not .get_ref())
        let get_inner_value = compiler.fun1("get_inner_value", |_ctx, outer: Var<Outer>| {
            // Navigate to inner field, then load its value
            outer.field(OuterType::inner()).get(InnerType::value())
        });

        let compiled = compiler
            .compile(get_inner_value)
            .expect("compilation failed");
        let f = compiled.as_fn();

        let test_struct = Outer {
            inner: Inner { value: 777 },
            extra: 123,
        };

        let result = f.call(test_struct); // Pass by value

        assert_eq!(result, 777);
    });
}

#[test]
fn test_nested_struct_by_ref_access() {
    for_each_backend(|mut compiler| {
        // fn get_inner_value_by_ref(outer: SRef<Outer>) -> i64
        // Access: outer.inner.value
        // Note: Outer is passed by REFERENCE
        let get_inner_value_by_ref =
            compiler.fun1("get_inner_value_by_ref", |_ctx, outer: Var<SRef<Outer>>| {
                // Get reference to inner field, then get its value
                // Note: Using get_ref (not get_ref_mut) since we have an immutable reference
                outer.get_ref(OuterType::inner()).get(InnerType::value())
            });

        let compiled = compiler
            .compile(get_inner_value_by_ref)
            .expect("compilation failed");
        let f = compiled.as_fn();

        let test_struct = Outer {
            inner: Inner { value: 555 },
            extra: 321,
        };
        let result = f.call(&test_struct); // Pass by reference

        assert_eq!(result, 555);
    });
}

#[test]
fn test_nested_struct_multiple_access() {
    for_each_backend(|mut compiler| {
        // fn sum_outer(outer: Outer) -> i64
        // Returns outer.inner.value + outer.extra
        // Note: Using .field() instead of .get_ref() since outer is by-value
        // (can't return references from by-value parameters)
        let sum_outer = compiler.fun1("sum_outer", |_ctx, outer: Var<Outer>| {
            let inner_val = outer.field(OuterType::inner()).get(InnerType::value());
            let extra = outer.get(OuterType::extra());

            inner_val + extra
        });

        let compiled = compiler.compile(sum_outer).expect("compilation failed");
        let f = compiled.as_fn();

        let test_struct = Outer {
            inner: Inner { value: 100 },
            extra: 50,
        };
        let result = f.call(test_struct); // Pass by value

        assert_eq!(result, 150); // 100 + 50
    });
}
#[test]
fn test_nested_struct_multiple_access_ref() {
    for_each_backend(|mut compiler| {
        // fn sum_outer(outer: Outer) -> i64
        // Returns outer.inner.value + outer.extra
        let sum_outer = compiler.fun1("sum_outer", |_ctx, outer: Var<SRef<Outer>>| {
            let inner_val = outer.get_ref(OuterType::inner()).get(InnerType::value());
            let extra = outer.get(OuterType::extra());

            inner_val + extra
        });

        let compiled = compiler.compile(sum_outer).expect("compilation failed");
        let f = compiled.as_fn();

        let test_struct = Outer {
            inner: Inner { value: 100 },
            extra: 50,
        };
        let result = f.call(&test_struct); // Pass by value

        assert_eq!(result, 150); // 100 + 50
    });
}

#[test]
fn test_struct_copy_semantics() {
    for_each_backend(|mut compiler| {
        // Test that Point is CopyType and passed by value
        // fn double_x(pt: Point) -> i64
        let double_x = compiler.fun1("double_x", |_ctx, pt: Var<Point>| {
            let x = pt.get(PointType::x());
            x * 2i64
        });

        let compiled = compiler.compile(double_x).expect("compilation failed");
        let f = compiled.as_fn();

        let point = Point { x: 21, y: 1.0 };
        let result = f.call(point); // Pass by value

        assert_eq!(result, 42);
    });
}

// Test accessing the second field of a nested struct's parent (to isolate the issue)
#[test]
fn test_outer_extra_field() {
    for_each_backend(|mut compiler| {
        // fn get_extra(outer: Outer) -> i64  -- just get the extra field, no nesting
        let get_extra = compiler.fun1("get_extra", |_ctx, outer: Var<Outer>| {
            outer.get(OuterType::extra())
        });

        let compiled = compiler.compile(get_extra).expect("compilation failed");
        let f = compiled.as_fn();

        let test_struct = Outer {
            inner: Inner { value: 777 },
            extra: 999,
        };
        let result = f.call(test_struct);

        assert_eq!(result, 999);
    });
}

// =============================================================================
// Tests for mixed integer/float struct ABI
// =============================================================================

#[test]
fn test_mixed_struct_read_f64_field() {
    for_each_backend(|mut compiler| {
        // fn get_y(pt: Point) -> f64 -- read the f64 field from mixed struct
        let get_y = compiler.fun1("get_y", |_ctx, pt: Var<Point>| pt.get(PointType::y()));

        let compiled = compiler.compile(get_y).expect("compilation failed");
        let f = compiled.as_fn();

        let point = Point { x: 42, y: 3.15 };
        let result = f.call(point);

        assert!(
            (result - 3.15).abs() < 1e-10,
            "Expected 3.15, got {}",
            result
        );
    });
}

#[test]
fn test_mixed_struct_read_i64_after_f64_access() {
    for_each_backend(|mut compiler| {
        // Read y (f64), then read x (i64) and return it
        // This verifies both fields are accessible
        let read_both = compiler.fun1("read_both", |_ctx, pt: Var<Point>| {
            let _y = pt.get(PointType::y()); // Access f64 field
            pt.get(PointType::x()) // Return i64 field
        });

        let compiled = compiler.compile(read_both).expect("compilation failed");
        let f = compiled.as_fn();

        let point = Point { x: 100, y: 3.15 };
        let result = f.call(point);

        assert_eq!(result, 100);
    });
}

#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct MixedStruct {
    #[staged(f64)]
    a: f64,
    #[staged(i64)]
    b: i64,
    #[staged(f64)]
    c: f64,
}

// 16-byte struct with f64 first - should work on ARM64
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct FloatFirst {
    #[staged(f64)]
    x: f64,
    #[staged(i64)]
    y: i64,
}

// These tests verify that large structs (>16 bytes) are correctly passed by pointer
#[test]
fn test_float_first_struct() {
    for_each_backend(|mut compiler| {
        // Struct with f64 as first field
        let get_a = compiler.fun1("get_a", |_ctx, s: Var<MixedStruct>| {
            s.get(MixedStructType::a())
        });

        let compiled = compiler.compile(get_a).expect("compilation failed");
        let f = compiled.as_fn();

        let s = MixedStruct {
            a: 2.71,
            b: 42,
            c: 1.41,
        };
        let result = f.call(s);

        assert!(
            (result - 2.71).abs() < 1e-10,
            "Expected 2.71, got {}",
            result
        );
    });
}

#[test]
fn test_float_first_struct_read_int() {
    for_each_backend(|mut compiler| {
        let get_b = compiler.fun1("get_b", |_ctx, s: Var<MixedStruct>| {
            s.get(MixedStructType::b())
        });

        let compiled = compiler.compile(get_b).expect("compilation failed");
        let f = compiled.as_fn();

        let s = MixedStruct {
            a: 2.71,
            b: 42,
            c: 1.41,
        };
        let result = f.call(s);

        assert_eq!(result, 42);
    });
}

// Test 16-byte struct with f64 first - should work because it fits in 2 registers
#[test]
fn test_16byte_float_first_struct() {
    for_each_backend(|mut compiler| {
        let get_x = compiler.fun1("get_x", |_ctx, s: Var<FloatFirst>| {
            s.get(FloatFirstType::x())
        });

        let compiled = compiler.compile(get_x).expect("compilation failed");
        let f = compiled.as_fn();

        let s = FloatFirst { x: 2.71, y: 42 };
        let result = f.call(s);

        assert!(
            (result - 2.71).abs() < 1e-10,
            "Expected 2.71, got {}",
            result
        );
    });
}

#[test]
fn test_16byte_float_first_struct_read_int() {
    for_each_backend(|mut compiler| {
        let get_y = compiler.fun1("get_y", |_ctx, s: Var<FloatFirst>| {
            s.get(FloatFirstType::y())
        });

        let compiled = compiler.compile(get_y).expect("compilation failed");
        let f = compiled.as_fn();

        let s = FloatFirst { x: 2.71, y: 42 };
        let result = f.call(s);

        assert_eq!(result, 42);
    });
}

// =============================================================================
// Generic structs: #[derive(StagedType)] over <A: StagedType, ...>
// =============================================================================

// The staged type of each field is *inferred* from its Rust type (no
// `#[staged(..)]`), since `A`/`B` are themselves `StagedType`.
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Pair<A: StagedType, B: StagedType> {
    first: A,
    second: B,
}

#[test]
fn test_generic_struct_by_value() {
    for_each_backend(|mut compiler| {
        // Pair<i64, i64>: the constructor fn's generics are inferred from the
        // receiver, so `PairType::second()` resolves to Field<Parent = Pair<i64,i64>>.
        let get_second = compiler.fun1("get_second", |_ctx, p: Var<Pair<i64, i64>>| {
            p.get(PairType::second())
        });

        let compiled = compiler.compile(get_second).expect("compilation failed");
        let f = compiled.as_fn();
        assert_eq!(
            f.call(Pair {
                first: 10i64,
                second: 20i64
            }),
            20
        );
    });
}

#[test]
fn test_generic_struct_distinct_monomorphizations() {
    for_each_backend(|mut compiler| {
        // A different instantiation: Pair<i32, i64>. `first` is an i32 at offset 0 —
        // exercises a per-monomorphization `offset_of!` and field type.
        let get_first = compiler.fun1("get_first", |_ctx, p: Var<Pair<i32, i64>>| {
            p.get(PairType::first())
        });

        let compiled = compiler.compile(get_first).expect("compilation failed");
        let f = compiled.as_fn();
        assert_eq!(
            f.call(Pair {
                first: 7i32,
                second: 99i64
            }),
            7
        );
    });
}

#[test]
fn test_generic_struct_by_ref() {
    for_each_backend(|mut compiler| {
        // By reference: get_ref into a field of a generic struct, then load it.
        let read_second = compiler.fun1("read_second", |_ctx, p: Var<SRef<Pair<i64, i64>>>| {
            load_ref(p.get_ref(PairType::second()))
        });

        let compiled = compiler.compile(read_second).expect("compilation failed");
        let f = compiled.as_fn();
        assert_eq!(
            f.call(&Pair {
                first: 1i64,
                second: 42i64
            }),
            42
        );
    });
}

#[test]
fn test_return_mixed_struct() {
    for_each_backend(|mut compiler| {
        // Function that takes two values and returns a Point struct
        let make_point = compiler.fun2("make_point", |_ctx, x: Var<i64>, _y: Var<f64>| {
            // We need to construct a Point - but we don't have struct construction yet
            // For now, just test that we can return the input x
            x
        });

        let compiled = compiler.compile(make_point).expect("compilation failed");
        let f = compiled.as_fn();

        let result = f.call(42, 3.15);
        assert_eq!(result, 42);
    });
}
