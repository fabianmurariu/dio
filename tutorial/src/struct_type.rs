//! Struct type support for staged computations.
//!
//! This module provides the infrastructure for working with `#[repr(C)]` structs
//! in JIT-compiled code. Since Cranelift doesn't have native aggregate type support,
//! we manually manage struct layout, alignment, and field offsets.
//!
//! # Design
//!
//! Structs are represented as:
//! - A `StructDef` describing the schema (fields, offsets, alignment)
//! - A pointer in JIT code that points to the struct data
//! - Field access via pointer arithmetic and loads
//!
//! # Usage
//!
//! ```ignore
//! use tutorial::struct_type::{StructDef, Field};
//! use tutorial::DataType;
//!
//! // Define struct schema at runtime
//! let edge_def = StructDef::new("Edge")
//!     .field("src", DataType::U64)
//!     .field("dst", DataType::U64)
//!     .build();
//!
//! // Or use the derive macro for compile-time structs
//! #[derive(StagedType)]
//! #[repr(C)]
//! struct Edge {
//!     src: u64,
//!     dst: u64,
//! }
//! ```

use std::sync::Arc;

use crate::num::PrimType;
use crate::DataType;

// =============================================================================
// FIELD DEFINITION
// =============================================================================

/// A field within a struct definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Field data type
    pub data_type: DataType,
    /// Byte offset from start of struct
    pub offset: usize,
}

impl Field {
    /// Create a new field (offset will be calculated by StructDef)
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Field {
            name: name.into(),
            data_type,
            offset: 0, // Set by StructDefBuilder
        }
    }

    /// Get the size of this field in bytes
    pub fn size(&self) -> usize {
        self.data_type.size_of()
    }

    /// Get the alignment requirement for this field
    pub fn alignment(&self) -> usize {
        self.data_type.align_of()
    }
}

// =============================================================================
// STRUCT DEFINITION
// =============================================================================

/// Definition of a struct type with computed layout.
///
/// This follows C struct layout rules (same as `#[repr(C)]`):
/// - Fields are laid out in declaration order
/// - Each field is aligned to its natural alignment
/// - Struct size is padded to a multiple of the largest alignment
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructDef {
    /// Struct name (for debugging and error messages)
    pub name: String,
    /// Fields with computed offsets
    pub fields: Arc<[Field]>,
    /// Total size in bytes (including trailing padding)
    pub size: usize,
    /// Alignment requirement (max of all field alignments)
    pub alignment: usize,
}

impl StructDef {
    /// Start building a new struct definition
    pub fn builder(name: impl Into<String>) -> StructDefBuilder {
        StructDefBuilder::new(name)
    }

    /// Create a struct definition from pre-computed fields
    ///
    /// This is used by the derive macro which computes offsets at compile time.
    pub fn from_fields(
        name: impl Into<String>,
        fields: Vec<Field>,
        size: usize,
        alignment: usize,
    ) -> Self {
        StructDef {
            name: name.into(),
            fields: fields.into(),
            size,
            alignment,
        }
    }

    /// Get a field by name
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Get a field by index
    pub fn field_by_index(&self, index: usize) -> Option<&Field> {
        self.fields.get(index)
    }

    /// Get the offset of a field by name
    pub fn field_offset(&self, name: &str) -> Option<usize> {
        self.field(name).map(|f| f.offset)
    }

    /// Get the number of fields
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Iterate over fields
    pub fn iter_fields(&self) -> impl Iterator<Item = &Field> {
        self.fields.iter()
    }

    /// Convert to DataType
    pub fn to_data_type(&self) -> DataType {
        DataType::Struct(Arc::new(self.clone()))
    }
}

// =============================================================================
// STRUCT DEFINITION BUILDER
// =============================================================================

/// Builder for constructing StructDef with automatic layout calculation.
pub struct StructDefBuilder {
    name: String,
    fields: Vec<(String, DataType)>,
}

impl StructDefBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        StructDefBuilder {
            name: name.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field to the struct
    pub fn field(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.fields.push((name.into(), data_type));
        self
    }

    /// Build the struct definition with computed layout
    ///
    /// This follows C struct layout rules:
    /// 1. Each field is placed at the next offset that satisfies its alignment
    /// 2. The struct's alignment is the maximum of all field alignments
    /// 3. The struct's size is padded to a multiple of its alignment
    pub fn build(self) -> StructDef {
        let mut computed_fields = Vec::with_capacity(self.fields.len());
        let mut current_offset = 0usize;
        let mut max_alignment = 1usize;

        for (name, data_type) in self.fields {
            let field_align = data_type.align_of();
            let field_size = data_type.size_of();

            // Align current offset to field alignment
            let padding = (field_align - (current_offset % field_align)) % field_align;
            current_offset += padding;

            computed_fields.push(Field {
                name,
                data_type,
                offset: current_offset,
            });

            current_offset += field_size;
            max_alignment = max_alignment.max(field_align);
        }

        // Pad struct size to alignment
        let final_padding = (max_alignment - (current_offset % max_alignment)) % max_alignment;
        let total_size = current_offset + final_padding;

        StructDef {
            name: self.name,
            fields: computed_fields.into(),
            size: total_size,
            alignment: max_alignment,
        }
    }
}

// =============================================================================
// DATATYPE SIZE AND ALIGNMENT
// =============================================================================

impl DataType {
    /// Get the size of this type in bytes
    pub fn size_of(&self) -> usize {
        match self {
            DataType::Prim(prim) => prim.size_of(),
            DataType::Bool => 1,
            DataType::Array { .. } => 16, // ptr (8) + len (8)
            DataType::ExtPtr(_) => 8,     // pointer size
            DataType::Unit => 0,
            DataType::Struct(def) => def.size,
            DataType::Slice { .. } => 16, // ptr (8) + len (8)
        }
    }

    /// Get the alignment requirement of this type in bytes
    pub fn align_of(&self) -> usize {
        match self {
            DataType::Prim(prim) => prim.align_of(),
            DataType::Bool => 1,
            DataType::Array { .. } => 8, // pointer alignment
            DataType::ExtPtr(_) => 8,    // pointer alignment
            DataType::Unit => 1,
            DataType::Struct(def) => def.alignment,
            DataType::Slice { .. } => 8, // pointer alignment
        }
    }
}

impl PrimType {
    /// Get the size of this primitive type in bytes
    pub fn size_of(self) -> usize {
        match self {
            PrimType::I8 | PrimType::U8 => 1,
            PrimType::I16 | PrimType::U16 => 2,
            PrimType::I32 | PrimType::U32 | PrimType::F32 => 4,
            PrimType::I64 | PrimType::U64 | PrimType::F64 => 8,
        }
    }

    /// Get the alignment requirement of this primitive type in bytes
    pub fn align_of(self) -> usize {
        // Natural alignment: alignment equals size
        self.size_of()
    }
}

// =============================================================================
// STAGED TYPE TRAIT
// =============================================================================

/// Trait for types that can be used in staged computations.
///
/// This trait provides the schema information needed to work with the type
/// in JIT code. It can be implemented manually or derived using `#[derive(StagedType)]`.
///
/// # Safety
///
/// Implementations must ensure that:
/// - The struct is `#[repr(C)]` (for predictable layout)
/// - Field offsets match the actual memory layout
/// - Size and alignment match `std::mem::size_of` and `std::mem::align_of`
pub trait StagedType: Sized {
    /// Get the struct definition for this type
    fn struct_def() -> &'static StructDef;

    /// Get the data type for this type
    fn data_type() -> DataType {
        DataType::Struct(Arc::new(Self::struct_def().clone()))
    }
}

// =============================================================================
// SLICE TYPE
// =============================================================================

/// Definition for a slice type `&[T]`.
///
/// Slices are represented as `(ptr: *const T, len: usize)` in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceDef {
    /// Element type
    pub element_type: DataType,
}

impl SliceDef {
    /// Create a new slice definition
    pub fn new(element_type: DataType) -> Self {
        SliceDef { element_type }
    }

    /// Get the element type
    pub fn element_type(&self) -> &DataType {
        &self.element_type
    }

    /// Convert to DataType
    pub fn to_data_type(&self) -> DataType {
        DataType::Slice {
            element_type: Box::new(self.element_type.clone()),
        }
    }
}

// =============================================================================
// STAGED STRUCT - Runtime struct pointer access
// =============================================================================

use cranelift_frontend::Variable;

use crate::expr::{Expr, StagedBuilder, Var};
use crate::num::StagedU64;

/// A staged struct value representing a pointer to struct data.
///
/// This type wraps a pointer to struct data and provides methods to
/// access fields. Field access generates JIT code that loads from
/// the appropriate memory offset.
///
/// # Example
///
/// ```ignore
/// let struct_ptr: StagedStruct = ...;
///
/// // Access field by name
/// let src = struct_ptr.field("src", &builder);  // Returns Expr
///
/// // Access field by index
/// let dst = struct_ptr.field_by_index(1, &builder);  // Returns Expr
/// ```
#[derive(Debug, Clone)]
pub struct StagedStruct {
    /// Variable holding the pointer to struct data
    ptr_var: Variable,
    /// Struct definition with field layout
    def: Arc<StructDef>,
}

impl StagedStruct {
    /// Create a new staged struct from a pointer variable and definition.
    pub fn new(ptr_var: Variable, def: Arc<StructDef>) -> Self {
        StagedStruct { ptr_var, def }
    }

    /// Get the struct definition.
    pub fn def(&self) -> &StructDef {
        &self.def
    }

    /// Get the pointer variable.
    pub fn ptr_var(&self) -> Variable {
        self.ptr_var
    }

    /// Get the data type of this struct.
    pub fn data_type(&self) -> DataType {
        DataType::Struct(self.def.clone())
    }

    /// Convert to an Expr (as ExtPtr)
    pub fn to_expr(&self) -> Expr {
        Expr::ExtPtr {
            var: self.ptr_var,
            name: self.def.name.clone(),
        }
    }

    /// Access a field by name.
    ///
    /// Generates JIT code that:
    /// 1. Takes the struct pointer
    /// 2. Adds the field offset
    /// 3. Loads the value at that address
    ///
    /// # Panics
    ///
    /// Panics if the field name doesn't exist.
    pub fn field(&self, name: &str, builder: &mut StagedBuilder) -> Expr {
        let field = self.def.field(name)
            .unwrap_or_else(|| panic!("Struct '{}' has no field '{}'", self.def.name, name));

        self.load_field(field, builder)
    }

    /// Access a field by index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn field_by_index(&self, index: usize, builder: &mut StagedBuilder) -> Expr {
        let field = self.def.field_by_index(index)
            .unwrap_or_else(|| panic!("Struct '{}' has no field at index {}", self.def.name, index));

        self.load_field(field, builder)
    }

    /// Load a field from the struct.
    fn load_field(&self, field: &Field, builder: &mut StagedBuilder) -> Expr {
        // ptr + offset
        let ptr = Var::new(self.ptr_var, DataType::ExtPtr(self.def.name.clone()));
        let offset = StagedU64::constant(field.offset as u64);

        // Generate: load(ptr + offset)
        builder.load_from_ptr(ptr, offset, field.data_type.clone())
    }

    /// Destructure into all fields.
    ///
    /// Returns a Vec of (name, Expr) pairs for all fields.
    pub fn destructure(&self, builder: &mut StagedBuilder) -> Vec<(String, Expr)> {
        self.def.fields.iter()
            .map(|field| (field.name.clone(), self.load_field(field, builder)))
            .collect()
    }

    /// Access a nested field by path.
    ///
    /// This allows accessing deeply nested struct fields with a single call.
    /// The path is an array of field names to traverse.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Instead of:
    /// // let start = line_struct.field("start", builder);
    /// // let x = start.as_struct().field("x", builder);
    ///
    /// // You can write:
    /// let x = line_struct.field_path(&["start", "x"], builder);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Any field name in the path doesn't exist
    /// - An intermediate field is not a struct type (except for the last field)
    /// - The path is empty
    pub fn field_path(&self, path: &[&str], builder: &mut StagedBuilder) -> Expr {
        assert!(!path.is_empty(), "field_path requires at least one field name");

        // Walk the path, accumulating offset and tracking current type
        let mut current_offset = 0u64;
        let mut current_def = self.def.clone();
        let mut final_type = None;

        for (i, &field_name) in path.iter().enumerate() {
            let field = current_def.field(field_name)
                .unwrap_or_else(|| panic!("Struct '{}' has no field '{}'", current_def.name, field_name));

            current_offset += field.offset as u64;
            final_type = Some(field.data_type.clone());

            // If not the last element, the field must be a struct to continue traversal
            if i < path.len() - 1 {
                match &field.data_type {
                    DataType::Struct(nested_def) => {
                        current_def = nested_def.clone();
                    }
                    _ => panic!(
                        "Field '{}' in struct '{}' is not a struct, cannot traverse further",
                        field_name, current_def.name
                    ),
                }
            }
        }

        // Generate single LoadField with accumulated offset
        let ptr = Var::new(self.ptr_var, DataType::ExtPtr(self.def.name.clone()));
        let offset = StagedU64::constant(current_offset);
        builder.load_from_ptr(ptr, offset, final_type.unwrap())
    }
}

/// A staged slice value representing a `&[T]` as (ptr, len).
///
/// Slices are represented as a pointer to the data and a length.
/// This type provides methods to access elements and iterate.
#[derive(Debug, Clone)]
pub struct StagedSlice {
    /// Variable holding the pointer to the first element
    ptr_var: Variable,
    /// Variable holding the length
    len_var: Variable,
    /// Element type
    element_type: DataType,
}

impl StagedSlice {
    /// Create a new staged slice from pointer and length variables.
    pub fn new(ptr_var: Variable, len_var: Variable, element_type: DataType) -> Self {
        StagedSlice {
            ptr_var,
            len_var,
            element_type,
        }
    }

    /// Get the element type.
    pub fn element_type(&self) -> &DataType {
        &self.element_type
    }

    /// Get the data type of this slice.
    pub fn data_type(&self) -> DataType {
        DataType::Slice {
            element_type: Box::new(self.element_type.clone()),
        }
    }

    /// Get the length as a StagedU64.
    pub fn len(&self) -> StagedU64 {
        StagedU64::variable(self.len_var)
    }

    /// Get an element at a given index.
    ///
    /// Generates JIT code that:
    /// 1. Computes ptr + index * element_size
    /// 2. Loads the value at that address
    pub fn get(&self, index: StagedU64, builder: &mut StagedBuilder) -> Expr {
        let ptr = Var::new(self.ptr_var, DataType::ExtPtr("slice_data".to_string()));
        let element_size = self.element_type.size_of() as u64;
        let offset = index * StagedU64::constant(element_size);

        builder.load_from_ptr(ptr, offset, self.element_type.clone())
    }

    /// Get the pointer variable.
    pub fn ptr_var(&self) -> Variable {
        self.ptr_var
    }

    /// Get the length variable.
    pub fn len_var(&self) -> Variable {
        self.len_var
    }
}

// =============================================================================
// STAGED BUILDER EXTENSIONS
// =============================================================================

impl StagedBuilder {
    /// Load a value from a pointer with an offset.
    ///
    /// Generates: load(ptr + offset) with the given data type.
    pub fn load_from_ptr(&mut self, ptr: Var, offset: StagedU64, data_type: DataType) -> Expr {
        Expr::LoadField {
            ptr: Box::new(ptr.to_expr()),
            offset: Box::new(Expr::U64(offset)),
            field_type: data_type,
        }
    }

    /// Store a value to a pointer with an offset.
    ///
    /// Generates: store(ptr + offset, value)
    pub fn store_to_ptr(&mut self, ptr: Var, offset: StagedU64, value: Expr) -> Expr {
        Expr::StoreField {
            ptr: Box::new(ptr.to_expr()),
            offset: Box::new(Expr::U64(offset)),
            value: Box::new(value),
        }
    }

    /// Create a StagedStruct from a pointer variable.
    pub fn struct_from_ptr(&self, ptr_var: Variable, def: Arc<StructDef>) -> StagedStruct {
        StagedStruct::new(ptr_var, def)
    }

    /// Create a StagedSlice from pointer and length variables.
    pub fn slice_from_vars(
        &self,
        ptr_var: Variable,
        len_var: Variable,
        element_type: DataType,
    ) -> StagedSlice {
        StagedSlice::new(ptr_var, len_var, element_type)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_struct_layout() {
        // struct Simple { a: u64, b: u64 }
        let def = StructDef::builder("Simple")
            .field("a", DataType::U64)
            .field("b", DataType::U64)
            .build();

        assert_eq!(def.name, "Simple");
        assert_eq!(def.field_count(), 2);
        assert_eq!(def.size, 16);
        assert_eq!(def.alignment, 8);

        let a = def.field("a").unwrap();
        assert_eq!(a.offset, 0);

        let b = def.field("b").unwrap();
        assert_eq!(b.offset, 8);
    }

    #[test]
    fn test_mixed_type_struct_layout() {
        // struct Mixed { a: u8, b: u64, c: u32 }
        // Layout: a(1) + padding(7) + b(8) + c(4) + padding(4) = 24
        let def = StructDef::builder("Mixed")
            .field("a", DataType::U8)
            .field("b", DataType::U64)
            .field("c", DataType::U32)
            .build();

        assert_eq!(def.alignment, 8);

        let a = def.field("a").unwrap();
        assert_eq!(a.offset, 0);

        let b = def.field("b").unwrap();
        assert_eq!(b.offset, 8); // aligned to 8

        let c = def.field("c").unwrap();
        assert_eq!(c.offset, 16); // after b

        // Total: 16 + 4 = 20, padded to 24 for 8-byte alignment
        assert_eq!(def.size, 24);
    }

    #[test]
    fn test_packed_struct_layout() {
        // struct Packed { a: u32, b: u32 }
        let def = StructDef::builder("Packed")
            .field("a", DataType::U32)
            .field("b", DataType::U32)
            .build();

        assert_eq!(def.size, 8);
        assert_eq!(def.alignment, 4);

        assert_eq!(def.field_offset("a"), Some(0));
        assert_eq!(def.field_offset("b"), Some(4));
    }

    #[test]
    fn test_nested_struct_layout() {
        // struct Inner { x: u64 }
        let inner_def = StructDef::builder("Inner")
            .field("x", DataType::U64)
            .build();

        // struct Outer { a: u8, inner: Inner, b: u32 }
        let outer_def = StructDef::builder("Outer")
            .field("a", DataType::U8)
            .field("inner", inner_def.to_data_type())
            .field("b", DataType::U32)
            .build();

        assert_eq!(outer_def.alignment, 8);

        let a = outer_def.field("a").unwrap();
        assert_eq!(a.offset, 0);

        let inner = outer_def.field("inner").unwrap();
        assert_eq!(inner.offset, 8); // aligned to 8 (Inner's alignment)

        let b = outer_def.field("b").unwrap();
        assert_eq!(b.offset, 16); // after Inner (8 bytes)

        // Total: 16 + 4 = 20, padded to 24
        assert_eq!(outer_def.size, 24);
    }

    #[test]
    fn test_field_by_index() {
        let def = StructDef::builder("Test")
            .field("first", DataType::I64)
            .field("second", DataType::U32)
            .build();

        let first = def.field_by_index(0).unwrap();
        assert_eq!(first.name, "first");

        let second = def.field_by_index(1).unwrap();
        assert_eq!(second.name, "second");

        assert!(def.field_by_index(2).is_none());
    }

    #[test]
    fn test_edge_tuple_struct() {
        // Simulating (usize, usize) as a struct
        let edge_def = StructDef::builder("Edge")
            .field("src", DataType::U64)
            .field("dst", DataType::U64)
            .build();

        assert_eq!(edge_def.size, 16);
        assert_eq!(edge_def.alignment, 8);
        assert_eq!(edge_def.field_offset("src"), Some(0));
        assert_eq!(edge_def.field_offset("dst"), Some(8));
    }

    #[test]
    fn test_slice_representation() {
        // Slice &[u64] is (ptr, len)
        let slice_def = SliceDef::new(DataType::U64);
        let dt = slice_def.to_data_type();

        assert_eq!(dt.size_of(), 16); // ptr + len
        assert_eq!(dt.align_of(), 8); // pointer alignment
    }

    #[test]
    fn test_bool_in_struct() {
        // struct WithBool { flag: bool, value: u64 }
        let def = StructDef::builder("WithBool")
            .field("flag", DataType::Bool)
            .field("value", DataType::U64)
            .build();

        assert_eq!(def.field_offset("flag"), Some(0));
        assert_eq!(def.field_offset("value"), Some(8)); // aligned to 8
        assert_eq!(def.size, 16);
    }

    #[test]
    fn test_verify_against_rust_layout() {
        // Verify our layout matches Rust's #[repr(C)] layout
        #[repr(C)]
        struct TestStruct {
            a: u8,
            b: u64,
            c: u32,
        }

        let def = StructDef::builder("TestStruct")
            .field("a", DataType::U8)
            .field("b", DataType::U64)
            .field("c", DataType::U32)
            .build();

        assert_eq!(def.size, std::mem::size_of::<TestStruct>());
        assert_eq!(def.alignment, std::mem::align_of::<TestStruct>());

        // Verify field offsets using memoffset-style calculation
        assert_eq!(def.field_offset("a"), Some(memoffset::offset_of!(TestStruct, a)));
        assert_eq!(def.field_offset("b"), Some(memoffset::offset_of!(TestStruct, b)));
        assert_eq!(def.field_offset("c"), Some(memoffset::offset_of!(TestStruct, c)));
    }

    // =========================================================================
    // JIT COMPILATION TESTS FOR STRUCT FIELD ACCESS
    // =========================================================================

    use crate::{Compiler, Expr, Ref};
    use crate::num::StagedU64;

    #[test]
    fn test_jit_struct_field_access() {
        // Test: read fields from a struct passed as a pointer

        // Define the struct: struct Edge { src: u64, dst: u64 }
        let edge_def = Arc::new(StructDef::builder("Edge")
            .field("src", DataType::U64)
            .field("dst", DataType::U64)
            .build());

        // Create actual struct data
        #[repr(C)]
        struct Edge {
            src: u64,
            dst: u64,
        }

        let edge = Edge { src: 42, dst: 99 };

        // Compile a function that takes a pointer to Edge and returns src + dst
        let mut compiler = Compiler::new().unwrap();
        let edge_def_clone = edge_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Edge".to_string())],
                DataType::U64,
                move |builder, vars| {
                    // Create StagedStruct from the pointer variable
                    let edge_struct = StagedStruct::new(vars[0], edge_def_clone.clone());

                    // Access fields
                    let src_expr = edge_struct.field("src", builder);
                    let dst_expr = edge_struct.field("dst", builder);

                    // Sum them
                    builder.let1(src_expr, |builder, src_var| {
                        builder.let1(dst_expr, |_, dst_var| {
                            let src = src_var.to_u64();
                            let dst = dst_var.to_u64();
                            Expr::U64(src + dst)
                        })
                    })
                },
            )
            .unwrap();

        // Call with reference to our edge struct using Ref wrapper
        let result = compiled.call([Ref::new(&edge)]).unwrap();

        assert_eq!(result.as_u64(), 141); // 42 + 99 = 141
    }

    #[test]
    fn test_jit_struct_field_access_mixed_types() {
        // Test: struct with mixed field types
        // Note: This test demonstrates loading fields of different sizes.
        // For now, we just return the u64 field to verify basic mixed-type struct works.
        // Full type-safe mixing (e.g., adding u32 + u64) would require explicit extend ops.

        // Define: struct Mixed { flag: u8, value: u64, count: u32 }
        let mixed_def = Arc::new(StructDef::builder("Mixed")
            .field("flag", DataType::U8)
            .field("value", DataType::U64)
            .field("count", DataType::U32)
            .build());

        // Create actual struct data
        #[repr(C)]
        struct Mixed {
            flag: u8,
            value: u64,
            count: u32,
        }

        let mixed = Mixed { flag: 1, value: 1000, count: 5 };

        // Verify the layout calculation matches repr(C)
        assert_eq!(mixed_def.field("flag").unwrap().offset, 0);
        assert_eq!(mixed_def.field("value").unwrap().offset, 8); // aligned to 8 bytes
        assert_eq!(mixed_def.field("count").unwrap().offset, 16); // after u64

        // Compile a function that just returns the value field (u64)
        let mut compiler = Compiler::new().unwrap();
        let mixed_def_clone = mixed_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Mixed".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let mixed_struct = StagedStruct::new(vars[0], mixed_def_clone.clone());

                    // Just access value (u64) - direct return without type conversion issues
                    let value_expr = mixed_struct.field("value", builder);

                    builder.let1(value_expr, |_, value_var| {
                        Expr::U64(value_var.to_u64())
                    })
                },
            )
            .unwrap();

        let result = compiled.call([Ref::new(&mixed)]).unwrap();

        assert_eq!(result.as_u64(), 1000);
    }

    #[test]
    fn test_jit_nested_struct_field_access() {
        // Test: nested struct access

        // Define: struct Point { x: u64, y: u64 }
        let point_def = Arc::new(StructDef::builder("Point")
            .field("x", DataType::U64)
            .field("y", DataType::U64)
            .build());

        // Define: struct Line { start: Point, end: Point }
        let line_def = Arc::new(StructDef::builder("Line")
            .field("start", DataType::Struct(point_def.clone()))
            .field("end", DataType::Struct(point_def.clone()))
            .build());

        // Create actual struct data
        #[repr(C)]
        struct Point {
            x: u64,
            y: u64,
        }

        #[repr(C)]
        struct Line {
            start: Point,
            end: Point,
        }

        let line = Line {
            start: Point { x: 10, y: 20 },
            end: Point { x: 30, y: 40 },
        };

        // Compile a function that returns start.x + end.y
        let mut compiler = Compiler::new().unwrap();
        let line_def_clone = line_def.clone();
        let point_def_clone = point_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Line".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let line_struct = StagedStruct::new(vars[0], line_def_clone.clone());

                    // For nested structs, we need to:
                    // 1. Get the address of the nested struct (ptr + offset)
                    // 2. Load fields from that address

                    // Get start.x: load from (line_ptr + 0 + 0)
                    // Get end.y: load from (line_ptr + 16 + 8)

                    // Access start field (which is a nested struct)
                    // start is at offset 0, x is at offset 0 within Point
                    let start_x_offset = StagedU64::constant(0); // start.x offset
                    let start_x = builder.load_from_ptr(
                        Var::new(vars[0], DataType::ExtPtr("Line".to_string())),
                        start_x_offset,
                        DataType::U64,
                    );

                    // end is at offset 16 (size of Point), y is at offset 8 within Point
                    let end_y_offset = StagedU64::constant(16 + 8); // end.y offset
                    let end_y = builder.load_from_ptr(
                        Var::new(vars[0], DataType::ExtPtr("Line".to_string())),
                        end_y_offset,
                        DataType::U64,
                    );

                    builder.let1(start_x, |builder, start_x_var| {
                        builder.let1(end_y, |_, end_y_var| {
                            let x = start_x_var.to_u64();
                            let y = end_y_var.to_u64();
                            Expr::U64(x + y)
                        })
                    })
                },
            )
            .unwrap();

        let result = compiled.call([Ref::new(&line)]).unwrap();

        assert_eq!(result.as_u64(), 50); // 10 + 40 = 50
    }

    #[test]
    fn test_derive_macro_simple() {
        use tutorial_derive::StagedType;

        // Simple struct with primitive fields
        #[derive(StagedType)]
        #[repr(C)]
        struct Edge {
            src: u64,
            dst: u64,
        }

        let def = Edge::struct_def();
        assert_eq!(def.name, "Edge");
        assert_eq!(def.fields.len(), 2);
        assert_eq!(def.field("src").unwrap().offset, 0);
        assert_eq!(def.field("dst").unwrap().offset, 8);
        assert_eq!(def.size, 16);
        assert_eq!(def.alignment, 8);
    }

    #[test]
    fn test_derive_macro_mixed_types() {
        use tutorial_derive::StagedType;

        #[derive(StagedType)]
        #[repr(C)]
        struct Record {
            flag: u8,
            id: u64,
            count: u32,
        }

        let def = Record::struct_def();
        assert_eq!(def.name, "Record");
        assert_eq!(def.fields.len(), 3);

        // Verify alignment/padding
        assert_eq!(def.field("flag").unwrap().offset, 0);
        assert_eq!(def.field("id").unwrap().offset, 8);    // aligned to 8 bytes
        assert_eq!(def.field("count").unwrap().offset, 16); // after u64
    }

    #[test]
    fn test_derive_macro_nested_structs() {
        use tutorial_derive::StagedType;

        #[derive(StagedType)]
        #[repr(C)]
        struct Point {
            x: u64,
            y: u64,
        }

        #[derive(StagedType)]
        #[repr(C)]
        struct Line {
            start: Point,
            end: Point,
        }

        let point_def = Point::struct_def();
        assert_eq!(point_def.size, 16);

        let line_def = Line::struct_def();
        assert_eq!(line_def.name, "Line");
        assert_eq!(line_def.fields.len(), 2);
        assert_eq!(line_def.field("start").unwrap().offset, 0);
        assert_eq!(line_def.field("end").unwrap().offset, 16);
        assert_eq!(line_def.size, 32);
    }

    #[test]
    fn test_derive_macro_jit_integration() {
        use tutorial_derive::StagedType;

        #[derive(StagedType)]
        #[repr(C)]
        struct Pair {
            a: u64,
            b: u64,
        }

        let pair = Pair { a: 100, b: 200 };
        let pair_def = Pair::struct_def();

        // Compile a function that returns a + b
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Pair".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let pair_struct = StagedStruct::new(vars[0], pair_def.clone());

                    let a_expr = pair_struct.field("a", builder);
                    let b_expr = pair_struct.field("b", builder);

                    builder.let1(a_expr, |builder, a_var| {
                        builder.let1(b_expr, |_, b_var| {
                            let a = a_var.to_u64();
                            let b = b_var.to_u64();
                            Expr::U64(a + b)
                        })
                    })
                },
            )
            .unwrap();

        use crate::Ref;
        let result = compiled.call([Ref::new(&pair)]).unwrap();

        assert_eq!(result.as_u64(), 300); // 100 + 200 = 300
    }

    #[test]
    fn test_slice_get_element() {
        // Test: slice element access
        // Pass slice as (ptr, len) and access elements by index

        let data: Vec<u64> = vec![10, 20, 30, 40, 50];
        let ptr = data.as_ptr() as u64;
        let len = data.len() as u64;

        // Compile a function that takes ptr and len, returns element at index 2
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64], // ptr, len
                DataType::U64,
                |builder, vars| {
                    let slice = StagedSlice::new(vars[0], vars[1], DataType::U64);

                    // Get element at index 2
                    let index = StagedU64::constant(2);
                    slice.get(index, builder)
                },
            )
            .unwrap();

        let result = compiled.call_u64(&[ptr, len]);
        assert_eq!(result, 30); // data[2] = 30
    }

    #[test]
    fn test_slice_sum_first_three() {
        // Test: sum first 3 elements in a slice (without loop, for simpler testing)
        // This verifies that slice element access at multiple indices works

        let data: Vec<u64> = vec![10, 20, 30, 40, 50];
        let ptr = data.as_ptr() as u64;
        let len = data.len() as u64;

        // Compile a function that returns data[0] + data[1] + data[2]
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64], // ptr, len
                DataType::U64,
                |builder, vars| {
                    let slice = StagedSlice::new(vars[0], vars[1], DataType::U64);

                    let elem0 = slice.get(StagedU64::constant(0), builder);
                    let elem1 = slice.get(StagedU64::constant(1), builder);
                    let elem2 = slice.get(StagedU64::constant(2), builder);

                    builder.let1(elem0, |builder, e0_var| {
                        builder.let1(elem1, |builder, e1_var| {
                            builder.let1(elem2, |_, e2_var| {
                                let sum = e0_var.to_u64() + e1_var.to_u64() + e2_var.to_u64();
                                Expr::U64(sum)
                            })
                        })
                    })
                },
            )
            .unwrap();

        let result = compiled.call_u64(&[ptr, len]);
        assert_eq!(result, 60); // 10 + 20 + 30 = 60
    }

    #[test]
    fn test_slice_def() {
        // Test: SliceDef creation and methods

        let slice_def = SliceDef::new(DataType::U64);
        assert_eq!(slice_def.element_type(), &DataType::U64);

        let data_type = slice_def.to_data_type();
        assert!(matches!(data_type, DataType::Slice { .. }));

        if let DataType::Slice { element_type } = data_type {
            assert_eq!(*element_type, DataType::U64);
        }
    }

    #[test]
    fn test_datatype_size_and_alignment() {
        // Test: size_of and align_of for various types

        assert_eq!(DataType::U8.size_of(), 1);
        assert_eq!(DataType::U16.size_of(), 2);
        assert_eq!(DataType::U32.size_of(), 4);
        assert_eq!(DataType::U64.size_of(), 8);
        assert_eq!(DataType::I64.size_of(), 8);
        assert_eq!(DataType::F64.size_of(), 8);
        assert_eq!(DataType::Bool.size_of(), 1);

        // Slice is ptr + len = 16 bytes
        let slice_type = DataType::Slice { element_type: Box::new(DataType::U64) };
        assert_eq!(slice_type.size_of(), 16);
        assert_eq!(slice_type.align_of(), 8);

        // ExtPtr is a pointer = 8 bytes
        assert_eq!(DataType::ExtPtr("test".to_string()).size_of(), 8);
        assert_eq!(DataType::ExtPtr("test".to_string()).align_of(), 8);
    }

    #[test]
    fn test_ref_wrapper_with_staged_struct() {
        // Test: Using Ref wrapper with StagedStruct for proper lifetime tracking
        // This demonstrates the new safe API for passing struct references

        // Define struct: Point { x: u64, y: u64 }
        let point_def = Arc::new(StructDef::builder("Point")
            .field("x", DataType::U64)
            .field("y", DataType::U64)
            .build());

        #[repr(C)]
        struct Point {
            x: u64,
            y: u64,
        }

        let point = Point { x: 100, y: 200 };

        // Compile function using StagedStruct field access API
        let compiler = Compiler::new().unwrap();
        let point_def_clone = point_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Point".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let point_struct = StagedStruct::new(vars[0], point_def_clone.clone());

                    let x_expr = point_struct.field("x", builder);
                    let y_expr = point_struct.field("y", builder);

                    builder.let1(x_expr, |builder, x_var| {
                        builder.let1(y_expr, |_, y_var| {
                            Expr::U64(x_var.to_u64() + y_var.to_u64())
                        })
                    })
                },
            )
            .unwrap();

        // Call with Ref wrapper - properly tracks lifetime
        use crate::Ref;
        let result = compiled.call([Ref::new(&point)]).unwrap();
        assert_eq!(result.as_u64(), 300); // 100 + 200 = 300
    }

    #[test]
    fn test_refmut_wrapper_with_staged_struct() {
        // Test: Using RefMut wrapper with StagedStruct
        // This verifies that mutable references work correctly

        let edge_def = Arc::new(StructDef::builder("Edge")
            .field("src", DataType::U64)
            .field("dst", DataType::U64)
            .build());

        #[repr(C)]
        struct Edge {
            src: u64,
            dst: u64,
        }

        let mut edge = Edge { src: 42, dst: 58 };

        // Compile function to read from struct
        let compiler = Compiler::new().unwrap();
        let edge_def_clone = edge_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Edge".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let edge_struct = StagedStruct::new(vars[0], edge_def_clone.clone());

                    let src_expr = edge_struct.field("src", builder);
                    let dst_expr = edge_struct.field("dst", builder);

                    builder.let1(src_expr, |builder, src_var| {
                        builder.let1(dst_expr, |_, dst_var| {
                            Expr::U64(src_var.to_u64() + dst_var.to_u64())
                        })
                    })
                },
            )
            .unwrap();

        // Call with RefMut wrapper - lifetime tracking prevents use-after-free
        use crate::RefMut;
        let result = compiled.call([RefMut::new(&mut edge)]).unwrap();
        assert_eq!(result.as_u64(), 100); // 42 + 58 = 100
    }

    #[test]
    fn test_field_path_nested_access() {
        // Test: field_path for accessing nested struct fields
        // This tests the new API that replaces .field().as_struct().field()

        // Define: struct Point { x: u64, y: u64 }
        let point_def = Arc::new(StructDef::builder("Point")
            .field("x", DataType::U64)
            .field("y", DataType::U64)
            .build());

        // Define: struct Line { start: Point, end: Point }
        let line_def = Arc::new(StructDef::builder("Line")
            .field("start", DataType::Struct(point_def.clone()))
            .field("end", DataType::Struct(point_def.clone()))
            .build());

        #[repr(C)]
        struct Point {
            x: u64,
            y: u64,
        }

        #[repr(C)]
        struct Line {
            start: Point,
            end: Point,
        }

        let line = Line {
            start: Point { x: 10, y: 20 },
            end: Point { x: 30, y: 40 },
        };

        // Compile function using field_path to access nested fields
        let compiler = Compiler::new().unwrap();
        let line_def_clone = line_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Line".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let line_struct = StagedStruct::new(vars[0], line_def_clone.clone());

                    // Access nested fields using field_path
                    let start_x = line_struct.field_path(&["start", "x"], builder);
                    let end_y = line_struct.field_path(&["end", "y"], builder);

                    builder.let1(start_x, |builder, start_x_var| {
                        builder.let1(end_y, |_, end_y_var| {
                            let x = start_x_var.to_u64();
                            let y = end_y_var.to_u64();
                            Expr::U64(x + y)
                        })
                    })
                },
            )
            .unwrap();

        let result = compiled.call([Ref::new(&line)]).unwrap();
        assert_eq!(result.as_u64(), 50); // 10 + 40 = 50
    }

    #[test]
    fn test_field_path_mixed_sizes() {
        // Test: field_path with different sized types (u8, i16, u32, u64, bool)
        // This verifies that loading smaller types from structs works correctly

        // Define: struct Inner { flag: bool, count: u32 }
        let inner_def = Arc::new(StructDef::builder("Inner")
            .field("flag", DataType::Bool)
            .field("count", DataType::U32)
            .build());

        // Define: struct Outer { id: u8, code: i16, data: Inner, value: u64 }
        let outer_def = Arc::new(StructDef::builder("Outer")
            .field("id", DataType::U8)
            .field("code", DataType::I16)
            .field("data", DataType::Struct(inner_def.clone()))
            .field("value", DataType::U64)
            .build());

        #[repr(C)]
        struct Inner {
            flag: bool,
            count: u32,
        }

        #[repr(C)]
        struct Outer {
            id: u8,
            code: i16,
            data: Inner,
            value: u64,
        }

        let outer = Outer {
            id: 5,
            code: -100,
            data: Inner { flag: true, count: 42 },
            value: 1000,
        };

        // Test 1: Access nested u32 field
        let compiler = Compiler::new().unwrap();
        let outer_def_clone = outer_def.clone();
        let mut compiled_u32 = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Outer".to_string())],
                DataType::U32,
                move |builder, vars| {
                    let outer_struct = StagedStruct::new(vars[0], outer_def_clone.clone());
                    let count = outer_struct.field_path(&["data", "count"], builder);
                    count // Returns U32
                },
            )
            .unwrap();

        let result = compiled_u32.call([Ref::new(&outer)]).unwrap();
        assert_eq!(result.as_u64() as u32, 42);

        // Test 2: Access nested bool field
        let compiler = Compiler::new().unwrap();
        let outer_def_clone = outer_def.clone();
        let mut compiled_bool = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Outer".to_string())],
                DataType::Bool,
                move |builder, vars| {
                    let outer_struct = StagedStruct::new(vars[0], outer_def_clone.clone());
                    let flag = outer_struct.field_path(&["data", "flag"], builder);
                    flag // Returns Bool
                },
            )
            .unwrap();

        let result = compiled_bool.call([Ref::new(&outer)]).unwrap();
        assert_eq!(result.as_bool(), true);

        // Test 3: Access top-level u8 field
        let compiler = Compiler::new().unwrap();
        let outer_def_clone = outer_def.clone();
        let mut compiled_u8 = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Outer".to_string())],
                DataType::U8,
                move |builder, vars| {
                    let outer_struct = StagedStruct::new(vars[0], outer_def_clone.clone());
                    outer_struct.field_path(&["id"], builder)
                },
            )
            .unwrap();

        let result = compiled_u8.call([Ref::new(&outer)]).unwrap();
        assert_eq!(result.as_u64() as u8, 5);

        // Test 4: Access top-level i16 field
        let compiler = Compiler::new().unwrap();
        let outer_def_clone = outer_def.clone();
        let mut compiled_i16 = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Outer".to_string())],
                DataType::I16,
                move |builder, vars| {
                    let outer_struct = StagedStruct::new(vars[0], outer_def_clone.clone());
                    outer_struct.field_path(&["code"], builder)
                },
            )
            .unwrap();

        let result = compiled_i16.call([Ref::new(&outer)]).unwrap();
        assert_eq!(result.as_i64() as i16, -100);
    }

    #[test]
    fn test_top_level_small_types() {
        // Test: Passing small types (u8, i16, u32, bool) as top-level arguments
        // This verifies that the argument passing mechanism handles small types correctly

        // Test 1: u8 argument
        let compiler = Compiler::new().unwrap();
        let mut compiled_u8 = compiler
            .compile_nary(
                vec![DataType::U8],
                DataType::U8,
                |_builder, vars| {
                    Expr::variable(vars[0], DataType::U8)
                },
            )
            .unwrap();

        let result = compiled_u8.call([42u8 as u64]).unwrap();
        assert_eq!(result.as_u64() as u8, 42);

        // Test 2: i16 argument
        let compiler = Compiler::new().unwrap();
        let mut compiled_i16 = compiler
            .compile_nary(
                vec![DataType::I16],
                DataType::I16,
                |_builder, vars| {
                    Expr::variable(vars[0], DataType::I16)
                },
            )
            .unwrap();

        let result = compiled_i16.call([(-100i16 as u64)]).unwrap();
        assert_eq!(result.as_i64() as i16, -100);

        // Test 3: u32 argument
        let compiler = Compiler::new().unwrap();
        let mut compiled_u32 = compiler
            .compile_nary(
                vec![DataType::U32],
                DataType::U32,
                |_builder, vars| {
                    Expr::variable(vars[0], DataType::U32)
                },
            )
            .unwrap();

        let result = compiled_u32.call([12345u32 as u64]).unwrap();
        assert_eq!(result.as_u64() as u32, 12345);

        // Test 4: bool argument
        let compiler = Compiler::new().unwrap();
        let mut compiled_bool = compiler
            .compile_nary(
                vec![DataType::Bool],
                DataType::Bool,
                |_builder, vars| {
                    Expr::variable(vars[0], DataType::Bool)
                },
            )
            .unwrap();

        let result = compiled_bool.call([1u64]).unwrap(); // true
        assert_eq!(result.as_bool(), true);

        let result = compiled_bool.call([0u64]).unwrap(); // false
        assert_eq!(result.as_bool(), false);
    }

    #[test]
    fn test_deeply_nested_field_path() {
        // Test: field_path with 3 levels of nesting
        // This ensures field_path works for arbitrary nesting depth

        // Define: struct Point { x: u64, y: u64 }
        let point_def = Arc::new(StructDef::builder("Point")
            .field("x", DataType::U64)
            .field("y", DataType::U64)
            .build());

        // Define: struct Box { min: Point, max: Point }
        let box_def = Arc::new(StructDef::builder("Box")
            .field("min", DataType::Struct(point_def.clone()))
            .field("max", DataType::Struct(point_def.clone()))
            .build());

        // Define: struct Scene { bounds: Box, id: u64 }
        let scene_def = Arc::new(StructDef::builder("Scene")
            .field("bounds", DataType::Struct(box_def.clone()))
            .field("id", DataType::U64)
            .build());

        #[repr(C)]
        struct Point {
            x: u64,
            y: u64,
        }

        #[repr(C)]
        struct Box {
            min: Point,
            max: Point,
        }

        #[repr(C)]
        struct Scene {
            bounds: Box,
            id: u64,
        }

        let scene = Scene {
            bounds: Box {
                min: Point { x: 0, y: 0 },
                max: Point { x: 100, y: 200 },
            },
            id: 42,
        };

        // Access scene.bounds.max.y (3 levels deep)
        let compiler = Compiler::new().unwrap();
        let scene_def_clone = scene_def.clone();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::ExtPtr("Scene".to_string())],
                DataType::U64,
                move |builder, vars| {
                    let scene_struct = StagedStruct::new(vars[0], scene_def_clone.clone());
                    scene_struct.field_path(&["bounds", "max", "y"], builder)
                },
            )
            .unwrap();

        let result = compiled.call([Ref::new(&scene)]).unwrap();
        assert_eq!(result.as_u64(), 200);
    }
}
