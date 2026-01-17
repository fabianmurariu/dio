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
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, FnArg, ItemFn, ReturnType, Type};

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
            "StagedType can only be derived for structs with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    // Only support named structs
    let struct_data = match &input.data {
        Data::Struct(s) => s,
        _ => {
            return syn::Error::new_spanned(&input, "StagedType can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let named_fields = match &struct_data.fields {
        Fields::Named(fields) => &fields.named,
        _ => {
            return syn::Error::new_spanned(
                &input,
                "StagedType can only be derived for structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    let struct_name = &input.ident;
    let _vis = &input.vis;

    // Generate field token module name
    let field_module_name = syn::Ident::new(&format!("{}Type", struct_name), struct_name.span());

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
                .expect(&format!(
                    "Field '{}' must have a #[staged(Type)] attribute",
                    field_name
                ));

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
            type RuntimeValue = #struct_name;

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

// =============================================================================
// #[extern_fn] - Attribute macro for external functions
// =============================================================================

/// Convert a Rust type to its corresponding StagedType marker.
///
/// Returns a token stream that evaluates to the staged type.
fn rust_type_to_staged_type(ty: &Type) -> Result<proc_macro2::TokenStream, String> {
    match ty {
        Type::Path(type_path) => {
            let path = &type_path.path;

            // Handle primitive types
            if path.is_ident("i8") {
                return Ok(quote! { ::rust_lms::types::I8Type });
            }
            if path.is_ident("u8") {
                return Ok(quote! { ::rust_lms::types::U8Type });
            }
            if path.is_ident("i16") {
                return Ok(quote! { ::rust_lms::types::I16Type });
            }
            if path.is_ident("u16") {
                return Ok(quote! { ::rust_lms::types::U16Type });
            }
            if path.is_ident("i32") {
                return Ok(quote! { ::rust_lms::types::I32Type });
            }
            if path.is_ident("u32") {
                return Ok(quote! { ::rust_lms::types::U32Type });
            }
            if path.is_ident("i64") {
                return Ok(quote! { ::rust_lms::types::I64Type });
            }
            if path.is_ident("u64") {
                return Ok(quote! { ::rust_lms::types::U64Type });
            }
            if path.is_ident("f32") {
                return Ok(quote! { ::rust_lms::types::F32Type });
            }
            if path.is_ident("f64") {
                return Ok(quote! { ::rust_lms::types::F64Type });
            }
            if path.is_ident("bool") {
                return Ok(quote! { ::rust_lms::types::BoolType });
            }
            if path.is_ident("usize") {
                return Ok(quote! { ::rust_lms::types::U64Type });
            }
            if path.is_ident("isize") {
                return Ok(quote! { ::rust_lms::types::I64Type });
            }

            // Handle generic types like COption<T>, FatSlice<T>
            if let Some(segment) = path.segments.last() {
                let ident = &segment.ident;

                if ident == "COption" {
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            let inner_staged = rust_type_to_staged_type(inner_ty)?;
                            return Ok(quote! { ::rust_lms::option::COptionType<#inner_staged> });
                        }
                    }
                    return Err("COption requires a type argument".to_string());
                }

                if ident == "FatSlice" {
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            let inner_staged = rust_type_to_staged_type(inner_ty)?;
                            return Ok(quote! { ::rust_lms::ffi::FatSliceType<#inner_staged> });
                        }
                    }
                    return Err("FatSlice requires a type argument".to_string());
                }

                if ident == "FatSliceMut" {
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            let inner_staged = rust_type_to_staged_type(inner_ty)?;
                            return Ok(quote! { ::rust_lms::ffi::FatSliceMutType<#inner_staged> });
                        }
                    }
                    return Err("FatSliceMut requires a type argument".to_string());
                }

                // Assume it's a custom struct type that implements StagedType
                return Ok(quote! { #path });
            }

            Err(format!("Unsupported type: {:?}", ty))
        }
        Type::Ptr(type_ptr) => {
            let elem_ty = &*type_ptr.elem;
            let inner_staged = rust_type_to_staged_type(elem_ty)?;

            if type_ptr.mutability.is_some() {
                // *mut T -> SMutPtr<T>
                Ok(quote! { ::rust_lms::refer::SMutPtr<#inner_staged> })
            } else {
                // *const T -> SPtr<T>
                Ok(quote! { ::rust_lms::refer::SPtr<#inner_staged> })
            }
        }
        Type::Reference(type_ref) => {
            let elem_ty = &*type_ref.elem;

            // Check for slice types &[T] or &mut [T]
            if let Type::Slice(slice_ty) = elem_ty {
                let inner_ty = &*slice_ty.elem;
                let inner_staged = rust_type_to_staged_type(inner_ty)?;

                if type_ref.mutability.is_some() {
                    // &mut [T] -> FatSliceMutType<T>
                    return Ok(quote! { ::rust_lms::ffi::FatSliceMutType<#inner_staged> });
                } else {
                    // &[T] -> FatSliceType<T>
                    return Ok(quote! { ::rust_lms::ffi::FatSliceType<#inner_staged> });
                }
            }

            let inner_staged = rust_type_to_staged_type(elem_ty)?;

            if type_ref.mutability.is_some() {
                // &mut T -> SRefMut<T>
                Ok(quote! { ::rust_lms::refer::SRefMut<#inner_staged> })
            } else {
                // &T -> SRef<T>
                Ok(quote! { ::rust_lms::refer::SRef<#inner_staged> })
            }
        }
        Type::Tuple(tuple) if tuple.elems.is_empty() => {
            // () -> UnitType
            Ok(quote! { ::rust_lms::types::UnitType })
        }
        _ => Err(format!(
            "Unsupported type for extern function: {:?}. \
            Supported types: primitives (i64, u64, f64, etc.), \
            pointers (*const T, *mut T), references (&T, &mut T), \
            slices (&[T], &mut [T]), COption<T>, FatSlice<T>, FatSliceMut<T>, \
            and #[repr(C)] structs with derive(StagedType)",
            ty
        )),
    }
}

/// Attribute macro for external `extern "C"` functions.
///
/// This macro generates a type that implements `ExternFn`, which provides
/// all metadata needed to register and call the function from JIT code.
///
/// # Requirements
///
/// - The function MUST be `extern "C"`
/// - The function SHOULD have `#[no_mangle]` to ensure the symbol is exported
/// - All parameter and return types must be FFI-safe
///
/// # Supported Types
///
/// - Primitives: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`, `f64`, `bool`
/// - Pointers: `*const T`, `*mut T`
/// - References: `&T`, `&mut T` (for FFI-safe T)
/// - Slices: `&[T]`, `&mut [T]` (converted to FatSlice/FatSliceMut)
/// - Option: `COption<T>`
/// - Fat slices: `FatSlice<T>`, `FatSliceMut<T>`
/// - Custom structs: Any `#[repr(C)]` struct with `derive(StagedType)`
///
/// # Example
///
/// ```ignore
/// use rust_lms_derive::extern_fn;
/// use rust_lms::ffi::FatSlice;
///
/// #[extern_fn]
/// #[no_mangle]
/// pub extern "C" fn sum_array(data: FatSlice<i64>) -> i64 {
///     unsafe { data.as_slice().iter().sum() }
/// }
///
/// // This generates:
/// // pub struct SumArrayExtern;
/// // unsafe impl ExternFn for SumArrayExtern { ... }
///
/// // Usage:
/// // let sum_fn = compiler.extern_fn::<SumArrayExtern>();
/// // let result = call_extern1::<_, _, FatSliceType<I64Type>, I64Type>(sum_fn, slice_arg);
/// ```
#[proc_macro_attribute]
pub fn extern_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    // Verify it's extern "C"
    let is_extern_c = input
        .sig
        .abi
        .as_ref()
        .and_then(|abi| abi.name.as_ref())
        .map(|name| name.value() == "C")
        .unwrap_or(false);

    if !is_extern_c {
        return syn::Error::new_spanned(
            &input.sig.abi,
            "#[extern_fn] can only be used on extern \"C\" functions",
        )
        .to_compile_error()
        .into();
    }

    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Generate the type name: my_function -> MyFunctionExtern
    let type_name = {
        let chars = fn_name_str.chars();
        let mut result = String::new();
        let mut capitalize_next = true;

        for c in chars {
            if c == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(c.to_ascii_uppercase());
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }

        format_ident!("{}Extern", result)
    };

    // Extract parameter types
    let mut param_staged_types = Vec::new();
    let mut errors = Vec::new();

    for input_arg in &input.sig.inputs {
        match input_arg {
            FnArg::Typed(pat_type) => {
                let ty = &*pat_type.ty;
                match rust_type_to_staged_type(ty) {
                    Ok(staged_type) => param_staged_types.push(staged_type),
                    Err(e) => errors.push(syn::Error::new_spanned(ty, e)),
                }
            }
            FnArg::Receiver(_) => {
                errors.push(syn::Error::new_spanned(
                    input_arg,
                    "extern \"C\" functions cannot have self parameters",
                ));
            }
        }
    }

    // Extract return type
    let return_staged_type = match &input.sig.output {
        ReturnType::Default => quote! { ::rust_lms::types::UnitType },
        ReturnType::Type(_, ty) => match rust_type_to_staged_type(ty) {
            Ok(staged_type) => staged_type,
            Err(e) => {
                errors.push(syn::Error::new_spanned(ty, e));
                quote! { ::rust_lms::types::UnitType }
            }
        },
    };

    // If there were any errors, return them
    if !errors.is_empty() {
        let compile_errors: proc_macro2::TokenStream = errors
            .into_iter()
            .map(|e| e.to_compile_error())
            .collect();
        return TokenStream::from(quote! {
            #input
            #compile_errors
        });
    }

    let num_params = param_staged_types.len();

    // Generate param_abi_types() implementation
    let param_abi_types_impl = if param_staged_types.is_empty() {
        quote! { Vec::new() }
    } else {
        quote! {
            vec![
                #(<#param_staged_types as ::rust_lms::types::StagedType>::abi_types()),*
            ]
        }
    };

    // Generate the output
    let expanded = quote! {
        #input

        /// Auto-generated extern function metadata type.
        ///
        /// Use with `compiler.extern_fn::<#type_name>()` to get a callable handle.
        #[allow(non_camel_case_types)]
        pub struct #type_name;

        unsafe impl ::rust_lms::ffi::ExternFn for #type_name {
            const NAME: &'static str = #fn_name_str;
            const NUM_PARAMS: usize = #num_params;
            const FN_PTR: *const u8 = #fn_name as *const u8;

            fn param_abi_types() -> Vec<Vec<::cranelift_codegen::ir::Type>> {
                #param_abi_types_impl
            }

            fn return_abi_types() -> Vec<::cranelift_codegen::ir::Type> {
                <#return_staged_type as ::rust_lms::types::StagedType>::abi_types()
            }
        }
    };

    TokenStream::from(expanded)
}
