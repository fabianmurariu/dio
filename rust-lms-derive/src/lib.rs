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
use syn::{
    parse_macro_input, parse_quote, Data, DeriveInput, Fields, FnArg, ItemFn, ReturnType, Type,
};

fn is_path(ty: &Type, expected: &str) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident(expected))
}

/// These mappings intentionally erase a Rust field to an integer with the same
/// bits. All other fields must use a staged marker whose `RuntimeValue` is the
/// field's actual Rust type.
fn is_supported_erased_field(field_ty: &Type, staged_ty: &Type) -> bool {
    (is_path(field_ty, "usize") && is_path(staged_ty, "u64"))
        || (is_path(field_ty, "isize") && is_path(staged_ty, "i64"))
        || (matches!(field_ty, Type::Ptr(_)) && is_path(staged_ty, "u64"))
}

fn is_marker_runtime_value(field_ty: &Type, staged_ty: &Type) -> bool {
    quote!(#field_ty).to_string() == quote!(#staged_ty :: RuntimeValue).to_string()
}

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
///     #[staged(i64)]
///     x: i64,
///     #[staged(f64)]
///     y: f64,
/// }
/// ```
///
/// This generates:
/// - `PointType` module with field accessors (`PointType::x`, `PointType::y`)
/// - `StagedType` impl with `RuntimeValue = Point` (owned, not reference)
/// - By-value `RuntimeParam` and `RuntimeResult` impls
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

    // Split the struct's generics so every generated impl/type carries them.
    // `<'a, M: StagedType>` -> impl_generics (with bounds), ty_generics (`<'a, M>`),
    // where_clause (the struct's own `where`, if any).
    let generics = &input.generics;
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    // A PhantomData tuple covering every generic param, so the (generic) field
    // marker structs use all of them (no "unused parameter" error). Const params
    // are out of scope and intentionally skipped.
    let phantom_elems: Vec<proc_macro2::TokenStream> = generics
        .params
        .iter()
        .filter_map(|p| match p {
            syn::GenericParam::Lifetime(l) => {
                let lt = &l.lifetime;
                Some(quote! { & #lt () })
            }
            syn::GenericParam::Type(t) => {
                let id = &t.ident;
                Some(quote! { #id })
            }
            syn::GenericParam::Const(_) => None,
        })
        .collect();
    let phantom_ty = quote! { ::core::marker::PhantomData<( #(#phantom_elems,)* )> };

    // Generate field token module name
    let field_module_name = format_ident!("{}Type", struct_name);

    // Resolve each field's staged type: explicit `#[staged(Ty)]`, else infer it
    // from the field's own Rust type (which must itself be a `StagedType`).
    let resolve_staged_ty = |field: &syn::Field| -> Type {
        field
            .attrs
            .iter()
            .find(|attr| attr.path().is_ident("staged"))
            .and_then(|attr| {
                attr.meta
                    .require_list()
                    .ok()
                    .and_then(|ml| syn::parse2::<Type>(ml.tokens.clone()).ok())
            })
            .unwrap_or_else(|| field.ty.clone())
    };

    // Per-field generated items: a marker type, its Copy/Clone + Field impls, and
    // a constructor fn named after the field (the public call surface,
    // `PointType::x()`), with generic params inferred from the receiver.
    let staged_types: Vec<Type> = named_fields.iter().map(&resolve_staged_ty).collect();

    // The derive itself emits unsafe trait implementations, so safe input must
    // prove the representation facts those traits require. Ordinary fields use
    // exact RuntimeValue equality. The small allowlist above covers the existing
    // pointer/usize-to-integer erasures and is checked by LAYOUT_VALID below.
    let mut trusted_generics = input.generics.clone();
    trusted_generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#struct_name #ty_generics: ::core::marker::Copy));
    for (field, staged_ty) in named_fields.iter().zip(&staged_types) {
        let field_ty = &field.ty;
        if !is_supported_erased_field(field_ty, staged_ty)
            && !is_marker_runtime_value(field_ty, staged_ty)
        {
            trusted_generics
                .make_where_clause()
                .predicates
                .push(parse_quote!(
                    #staged_ty: ::rust_lms::types::StagedType<RuntimeValue = #field_ty>
                ));
        }
    }
    let (trusted_impl_generics, _, trusted_where_clause) = trusted_generics.split_for_impl();

    let layout_checks = named_fields.iter().zip(&staged_types).map(|(field, staged_ty)| {
        let field_ty = &field.ty;
        quote! {
            assert!(
                ::core::mem::size_of::<#field_ty>()
                    == ::core::mem::size_of::<<#staged_ty as ::rust_lms::types::StagedType>::RuntimeValue>()
            );
            assert!(
                ::core::mem::align_of::<#field_ty>()
                    == ::core::mem::align_of::<<#staged_ty as ::rust_lms::types::StagedType>::RuntimeValue>()
            );
        }
    });

    let field_items = named_fields.iter().enumerate().map(|(idx, field)| {
        let field_name = field.ident.as_ref().unwrap();
        let marker = format_ident!("__field_{}", field_name);
        let staged_ty = resolve_staged_ty(field);

        // The marker is a pure `PhantomData` handle, so it needs no bounds and
        // no `where`; only the `Field` impl references the parent struct (as
        // `Parent` and in `offset_of!`), so the struct's `where` goes there.
        quote! {
            pub struct #marker #ty_generics ( #phantom_ty );

            impl #impl_generics ::core::clone::Clone for #marker #ty_generics {
                fn clone(&self) -> Self { *self }
            }
            impl #impl_generics ::core::marker::Copy for #marker #ty_generics {}

            unsafe impl #trusted_impl_generics ::rust_lms::_internal::Field for #marker #ty_generics #trusted_where_clause {
                type Parent = #struct_name #ty_generics;
                type Out = #staged_ty;
                const OFFSET: usize = {
                    let () = <#struct_name #ty_generics as ::rust_lms::types::StagedType>::LAYOUT_VALID;
                    ::core::mem::offset_of!(#struct_name #ty_generics, #field_name)
                };
                const INDEX: usize = #idx;
            }

            pub fn #field_name #impl_generics () -> #marker #ty_generics {
                #marker(::core::marker::PhantomData)
            }
        }
    });

    // CopyType is conditional: the struct is a staged `CopyType` exactly when all
    // of its field staged types are (merged with the struct's own `where`).
    let copy_where = {
        // `Self: Copy` makes the `CopyType: Copy` supertrait provable for
        // conditionally-`Copy` generic structs (where the inline bounds alone
        // don't imply it), and is trivially true for plain `Copy` structs.
        let mut preds: Vec<proc_macro2::TokenStream> = vec![quote! { Self: ::core::marker::Copy }];
        if let Some(wc) = trusted_where_clause {
            preds.extend(wc.predicates.iter().map(|p| quote! { #p }));
        }
        for ty in &staged_types {
            preds.push(quote! { #ty: ::rust_lms::types::CopyType });
        }
        quote! { where #(#preds),* }
    };

    // Generate the output
    let expanded = quote! {
        // Field token module. The markers are internal (`__field_*`) and the
        // module/marker names trip the case lints, so allow them here.
        #[allow(non_camel_case_types, non_snake_case)]
        pub mod #field_module_name {
            use super::*;

            #(#field_items)*
        }

        unsafe impl #trusted_impl_generics ::rust_lms::types::StagedType for #struct_name #ty_generics #trusted_where_clause {
            type RuntimeValue = #struct_name #ty_generics;

            const LAYOUT_VALID: () = {
                #(#layout_checks)*
            };

            fn cranelift_type() -> ::cranelift_codegen::ir::Type {
                // Internally we use I64 (pointer to stack slot)
                ::cranelift_codegen::ir::types::I64
            }

            fn size_of() -> usize {
                let () = Self::LAYOUT_VALID;
                ::std::mem::size_of::<#struct_name #ty_generics>()
            }

            fn align_of() -> usize {
                let () = Self::LAYOUT_VALID;
                ::std::mem::align_of::<#struct_name #ty_generics>()
            }

            fn is_copy_struct() -> bool {
                true
            }

            fn num_abi_values() -> usize {
                // Number of i64s needed to hold this struct (round up).
                ::std::mem::size_of::<#struct_name #ty_generics>().div_ceil(8)
            }

            fn abi_types() -> Vec<::cranelift_codegen::ir::Type> {
                // Return N x I64 where N = num_abi_values
                let n = ::std::mem::size_of::<#struct_name #ty_generics>().div_ceil(8);
                vec![::cranelift_codegen::ir::types::I64; n]
            }
        }

        unsafe impl #trusted_impl_generics ::rust_lms::types::RuntimeParam for #struct_name #ty_generics #trusted_where_clause {
            type Arg<'call> = #struct_name #ty_generics;
        }

        unsafe impl #trusted_impl_generics ::rust_lms::types::RuntimeResult for #struct_name #ty_generics #trusted_where_clause {
            type Output<'call> = #struct_name #ty_generics;
        }

        unsafe impl #trusted_impl_generics ::rust_lms::types::CopyType for #struct_name #ty_generics #copy_where {}
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
                return Ok(quote! { i8 });
            }
            if path.is_ident("u8") {
                return Ok(quote! { u8 });
            }
            if path.is_ident("i16") {
                return Ok(quote! { i16 });
            }
            if path.is_ident("u16") {
                return Ok(quote! { u16 });
            }
            if path.is_ident("i32") {
                return Ok(quote! { i32 });
            }
            if path.is_ident("u32") {
                return Ok(quote! { u32 });
            }
            if path.is_ident("i64") {
                return Ok(quote! { i64 });
            }
            if path.is_ident("u64") {
                return Ok(quote! { u64 });
            }
            if path.is_ident("f32") {
                return Ok(quote! { f32 });
            }
            if path.is_ident("f64") {
                return Ok(quote! { f64 });
            }
            if path.is_ident("bool") {
                return Ok(quote! { bool });
            }
            if path.is_ident("usize") {
                return Ok(quote! { u64 });
            }
            if path.is_ident("isize") {
                return Ok(quote! { i64 });
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

            // Preserve slice references as staged slice references. Their ABI
            // remains the same two-word `(ptr, len)` representation, but the
            // marker now retains Rust's shared/unique distinction.
            if let Type::Slice(slice_ty) = elem_ty {
                let inner_ty = &*slice_ty.elem;
                let inner_staged = rust_type_to_staged_type(inner_ty)?;

                if type_ref.mutability.is_some() {
                    return Ok(quote! {
                        ::rust_lms::refer::SRefMut<'static, ::rust_lms::slice::Slice<#inner_staged>>
                    });
                } else {
                    return Ok(quote! {
                        ::rust_lms::refer::SRef<'static, ::rust_lms::slice::Slice<#inner_staged>>
                    });
                }
            }

            // Generated code does not inspect an arbitrary referenced Rust
            // value, so retain its reference kind over an opaque pointee.
            if type_ref.mutability.is_some() {
                Ok(quote! {
                    ::rust_lms::refer::SRefMut<'static, ::rust_lms::opaque::Opaque<#elem_ty>>
                })
            } else {
                Ok(quote! {
                    ::rust_lms::refer::SRef<'static, ::rust_lms::opaque::Opaque<#elem_ty>>
                })
            }
        }
        Type::Tuple(tuple) if tuple.elems.is_empty() => {
            // () -> ()
            Ok(quote! { () })
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
/// - References: `&T`, `&mut T` (preserved as staged shared/unique references)
/// - Slices: `&[T]`, `&mut [T]` (preserved in metadata but not classified as
///   safe C ABI; prefer `FatSlice<T>` / `FatSliceMut<T>`)
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
/// // let result = call_extern1(sum_fn, slice_arg);
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
        ReturnType::Default => quote! { () },
        ReturnType::Type(_, ty) => match rust_type_to_staged_type(ty) {
            Ok(staged_type) => staged_type,
            Err(e) => {
                errors.push(syn::Error::new_spanned(ty, e));
                quote! { () }
            }
        },
    };

    // If there were any errors, return them
    if !errors.is_empty() {
        let compile_errors: proc_macro2::TokenStream =
            errors.into_iter().map(|e| e.to_compile_error()).collect();
        return TokenStream::from(quote! {
            #input
            #compile_errors
        });
    }

    let args_staged_type = if param_staged_types.is_empty() {
        quote! { () }
    } else {
        quote! { (#(#param_staged_types,)*) }
    };

    let has_reference_return = matches!(
        &input.sig.output,
        ReturnType::Type(_, ty) if matches!(&**ty, Type::Reference(_))
    );
    let has_slice_reference_param = input.sig.inputs.iter().any(|arg| {
        matches!(
            arg,
            FnArg::Typed(pat_type)
                if matches!(
                    &*pat_type.ty,
                    Type::Reference(reference) if matches!(&*reference.elem, Type::Slice(_))
                )
        )
    });

    // Reference parameters are safe once call_externN receives an SRef/SRefMut
    // and performs a staged reborrow. Reference results remain unchecked until
    // the staged result can carry provenance tied to the input call. Rust slice
    // references are also excluded because they are not a stable C ABI.
    let safe_extern_impl =
        if input.sig.unsafety.is_none() && !has_reference_return && !has_slice_reference_param {
            quote! {
                unsafe impl ::rust_lms::ffi::SafeExternFn for #type_name {}
            }
        } else {
            quote! {}
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
            type Args = #args_staged_type;
            type Ret = #return_staged_type;

            const NAME: &'static str = #fn_name_str;
            const FN_PTR: *const u8 = #fn_name as *const u8;

            fn return_abi_types() -> Vec<::cranelift_codegen::ir::Type> {
                <#return_staged_type as ::rust_lms::types::StagedType>::abi_types()
            }
        }

        #safe_extern_impl
    };

    TokenStream::from(expanded)
}
