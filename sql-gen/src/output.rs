//! Growable output columns, `SVec`-backed.
//!
//! Replaces the pre-sized `PreparedOutput`: a streaming query has no total row
//! count up front, so each output column is a *growable* host buffer the kernel
//! appends into. The fixed-width columns are [`SVec`]s (`rust-lms-std`), so the
//! per-value append is **inline JIT code** — `if len==cap { grow }; data[len]=v;
//! len++` — with the only FFI call ([`svec_grow`]) on the amortized-rare grow.
//! Strings stay on the existing `StringViewBuilder` append extern (variable
//! length). See `docs/table_scan.md` §4.
//!
//! The host owns [`OutCols`] (allocated before compile, alive across the run) and
//! hands the kernel an [`OutputHandle`] of baked, stable pointers — the same
//! "host outlives the kernel" contract as the string pool and the GROUP BY state.
//! Handle indirection is what makes this safe: the baked pointer targets a stable
//! [`RawVec`] control block, never the movable buffer, so a mid-run realloc never
//! dangles it.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringViewBuilder};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use rust_lms_std::{HostVec, RawVec};

/// A fixed-width value column's host storage, one growable buffer per physical type.
enum ValVec {
    I32(HostVec<i32>),
    I64(HostVec<i64>),
    F64(HostVec<f64>),
}

impl ValVec {
    fn new(dt: &DataType) -> Self {
        match dt {
            DataType::Int32 => ValVec::I32(HostVec::new()),
            DataType::Int64 => ValVec::I64(HostVec::new()),
            DataType::Float64 => ValVec::F64(HostVec::new()),
            other => panic!("unsupported output column type: {other}"),
        }
    }

    /// Stable control-block pointer to bake into the kernel.
    fn control_ptr(&mut self) -> *mut RawVec {
        match self {
            ValVec::I32(v) => v.as_raw_control_ptr(),
            ValVec::I64(v) => v.as_raw_control_ptr(),
            ValVec::F64(v) => v.as_raw_control_ptr(),
        }
    }

    /// Finalize the first `n` elements into an Arrow array (one copy; zero-copy via
    /// `Buffer::from_custom_allocation` is a later refinement — see `docs/table_scan.md`).
    fn into_array(self, n: usize, nulls: Option<NullBuffer>) -> ArrayRef {
        match self {
            ValVec::I32(v) => Arc::new(Int32Array::new(
                ScalarBuffer::from(v.as_slice()[..n].to_vec()),
                nulls,
            )),
            ValVec::I64(v) => Arc::new(Int64Array::new(
                ScalarBuffer::from(v.as_slice()[..n].to_vec()),
                nulls,
            )),
            ValVec::F64(v) => Arc::new(Float64Array::new(
                ScalarBuffer::from(v.as_slice()[..n].to_vec()),
                nulls,
            )),
        }
    }
}

/// One output column's host storage.
enum HostOutCol {
    /// Fixed-width value column. A nullable column pairs its value buffer with a
    /// `bool`-per-row validity buffer (`true` = valid); the kernel pushes both once
    /// per row, so they stay length-aligned.
    Fixed {
        values: ValVec,
        validity: Option<HostVec<bool>>,
    },
    /// `Utf8View` column: the existing Arrow builder (owns its own nulls). `Box`ed
    /// so its address is stable to bake regardless of the `cols` vec.
    Str { builder: Box<StringViewBuilder> },
}

/// Host-owned growable output: one column per output field, in schema order.
/// Allocated before the kernel is compiled; must outlive the JIT call (the baked
/// [`OutputHandle`] pointers reference its control blocks / builders).
pub struct OutCols {
    schema: SchemaRef,
    cols: Vec<HostOutCol>,
}

impl OutCols {
    /// Allocate an empty growable column per field (no buffer until the first push).
    pub fn alloc(schema: &SchemaRef) -> Self {
        let cols = schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                DataType::Utf8View => HostOutCol::Str {
                    builder: Box::new(StringViewBuilder::new()),
                },
                dt => HostOutCol::Fixed {
                    values: ValVec::new(dt),
                    validity: f.is_nullable().then(HostVec::<bool>::new),
                },
            })
            .collect();
        Self {
            schema: schema.clone(),
            cols,
        }
    }

    /// Baked, stable pointers handed to the kernel. Call once, after `alloc` and
    /// before compiling; the pointers stay valid until `self` is dropped.
    pub fn handle(&mut self) -> OutputHandle {
        let cols = self
            .cols
            .iter_mut()
            .map(|c| match c {
                HostOutCol::Str { builder } => OutColHandle::Str {
                    builder: &mut **builder as *mut StringViewBuilder,
                },
                HostOutCol::Fixed { values, validity } => OutColHandle::Fixed {
                    values: values.control_ptr(),
                    validity: validity.as_mut().map(|v| v.as_raw_control_ptr()),
                },
            })
            .collect();
        OutputHandle { cols }
    }

    /// Assemble the first `n` emitted rows into a `RecordBatch`.
    pub fn into_record_batch(self, n: usize) -> RecordBatch {
        let arrays = self
            .cols
            .into_iter()
            .map(|c| match c {
                HostOutCol::Str { mut builder } => Arc::new(builder.finish()) as ArrayRef,
                HostOutCol::Fixed { values, validity } => {
                    let nulls =
                        validity.map(|v| NullBuffer::from_iter(v.as_slice()[..n].iter().copied()));
                    values.into_array(n, nulls)
                }
            })
            .collect::<Vec<_>>();
        RecordBatch::try_new(self.schema, arrays).expect("output columns match schema")
    }
}

/// The kernel-side handle: one baked pointer set per output column. Consumed by
/// `codegen::write_col`, which reconstructs a typed [`SVec`](rust_lms_std::SVec)
/// (fixed) or an opaque `&mut StringViewBuilder` (string) from these addresses.
pub struct OutputHandle {
    pub cols: Vec<OutColHandle>,
}

/// Baked pointers for one output column. Real typed pointers (control block /
/// builder), not `u64`s — the pointee type is checked at stage 0 in codegen.
pub enum OutColHandle {
    /// Fixed-width: the value `SVec`'s control block, plus the validity `SVec`'s
    /// control block when the field is nullable.
    Fixed {
        values: *mut RawVec,
        validity: Option<*mut RawVec>,
    },
    /// `Utf8View`: the output `StringViewBuilder` to append into.
    Str { builder: *mut StringViewBuilder },
}
