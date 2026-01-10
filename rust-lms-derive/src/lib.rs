//! Proc macro for deriving StagedType on structs
//!
//! This derive macro generates:
//! - Field token types for accessing struct fields
//! - StagedType implementation with pass-by-value semantics
//! - CopyType implementation (if all fields are Copy)
//! - Field accessor methods
//!
//! # Pass-by-Value Semantics
//!
//! Structs derived with `StagedType` use pass-by-value semantics at the Rust ABI level:
//! - `Var<Point>` means `fn(Point)` - the struct is passed by value
//! - `Var<SRef<Point>>` means `fn(&Point)` - passed by reference
//! - `Var<SRefMut<Point>>` means `fn(&mut Point)` - passed by mutable reference
//!
//! Internally, structs are stored in stack slots and accessed via pointers for
//! field access, but this is transparent to the user.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

/// Derive macro for generating StagedType implementation for structs.
///
/// Requirements:
/// - The struct MUST be annotated with #[repr(C)]
/// - The struct MUST implement Copy (for pass-by-value semantics)
/// - Fields should be actual Rust types (i64, f64, etc.)
/// - Each field should be annotated with #[staged(TypeMarker)] to indicate its staged type
///
/// # Example
/// ```ignore
/// #[derive(StagedType, Copy, Clone)]
/// #[repr(C)]
/// struct Point {
///     #[staged(I64Type)]
///     x: i64,
///     #[staged(F64Type)]
///     y: f64,
/// }
/// ```
///
/// This generates:
/// - `PointType` module with field accessors (`PointType::x`, `PointType::y`)
/// - `StagedType` impl with `RuntimeValue<'a> = Point` (owned, not reference)
/// - `CopyType` impl
#[proc_macro_derive(StagedType, attributes(staged))]
pub fn derive_staged_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Check for repr(C)
    let has_repr_c = input.attrs.iter().any(|attr| {
        if attr.path().is_ident("repr") {
            if let Ok(meta_list) = attr.meta.require_list() {
                return meta_list.tokens.to_string() == "C";
            }
        }
        false
    });

    if !has_repr_c {
        return syn::Error::new_spanned(
            &input,
            "StagedType can only be derived for structs with #[repr(C)]"
        )
        .to_compile_error()
        .into();
    }

    // Only support named structs
    let struct_data = match &input.data {
        Data::Struct(s) => s,
        _ => {
            return syn::Error::new_spanned(
                &input,
                "StagedType can only be derived for structs"
            )
            .to_compile_error()
            .into();
        }
    };

    let named_fields = match &struct_data.fields {
        Fields::Named(fields) => &fields.named,
        _ => {
            return syn::Error::new_spanned(
                &input,
                "StagedType can only be derived for structs with named fields"
            )
            .to_compile_error()
            .into();
        }
    };

    let struct_name = &input.ident;
    let _vis = &input.vis;

    // Generate field token module name
    let field_module_name = syn::Ident::new(
        &format!("{}Type", struct_name),
        struct_name.span()
    );

    // Collect field information
    let field_tokens: Vec<_> = named_fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            let field_name = field.ident.as_ref().unwrap();
            let _field_ty = &field.ty;

            // Extract staged type from #[staged(Type)] attribute
            let staged_ty = field
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("staged"))
                .and_then(|attr| {
                    if let Ok(meta_list) = attr.meta.require_list() {
                        let tokens = &meta_list.tokens;
                        syn::parse2::<Type>(tokens.clone()).ok()
                    } else {
                        None
                    }
                })
                .expect(&format!("Field '{}' must have a #[staged(Type)] attribute", field_name));

            // Calculate offset
            let offset_calc = quote! {
                memoffset::offset_of!(#struct_name, #field_name)
            };

            quote! {
                #[derive(Copy, Clone)]
                pub struct #field_name;

                impl ::rust_lms::_internal::Field for #field_name {
                    type Parent = #struct_name;
                    type Out = #staged_ty;
                    const OFFSET: usize = #offset_calc;
                    const INDEX: usize = #idx;
                }
            }
        })
        .collect();

    // Extract staged types for CopyType check
    let staged_types: Vec<_> = named_fields
        .iter()
        .map(|field| {
            field
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("staged"))
                .and_then(|attr| {
                    if let Ok(meta_list) = attr.meta.require_list() {
                        let tokens = &meta_list.tokens;
                        syn::parse2::<Type>(tokens.clone()).ok()
                    } else {
                        None
                    }
                })
                .expect("Field must have #[staged(Type)] attribute")
        })
        .collect();

    // Generate the output
    let expanded = quote! {
        // Field token module
        pub mod #field_module_name {
            use super::*;

            #(#field_tokens)*
        }

        impl ::rust_lms::types::StagedType for #struct_name {
            type RuntimeValue<'a> = #struct_name;

            fn cranelift_type() -> ::cranelift_codegen::ir::Type {
                // Internally we use I64 (pointer to stack slot)
                ::cranelift_codegen::ir::types::I64
            }

            fn size_of() -> usize {
                ::std::mem::size_of::<#struct_name>()
            }

            fn align_of() -> usize {
                ::std::mem::align_of::<#struct_name>()
            }

            fn is_copy_struct() -> bool {
                true
            }

            fn num_abi_values() -> usize {
                // Number of i64s needed to hold this struct
                // Round up: (size + 7) / 8
                (::std::mem::size_of::<#struct_name>() + 7) / 8
            }

            fn abi_types() -> Vec<::cranelift_codegen::ir::Type> {
                // Return N x I64 where N = num_abi_values
                let n = (::std::mem::size_of::<#struct_name>() + 7) / 8;
                vec![::cranelift_codegen::ir::types::I64; n]
            }
        }

        // CopyType implementation - generated if all fields are CopyType
        // We use a const check to avoid compile errors
        const _: () = {
            // Helper function to check if all fields implement CopyType
            fn __check_copy_fields<T>()
            where
                #(#staged_types: ::rust_lms::types::CopyType,)*
            {}

            // If this compiles, all fields are CopyType
            impl ::rust_lms::types::CopyType for #struct_name {}
        };
    };

    TokenStream::from(expanded)
}
