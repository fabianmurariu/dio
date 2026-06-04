//! FunDef construction helpers for function definitions.

use crate::func::{Ctx, FunDef};
use crate::func_impl::{
    FunRef0, FunRef1, FunRef2, FunRef3, FunRef4, FunRef5, FunRef6, FunRef7, FunRef8, TypeInfo,
};
use crate::staged::{Staged, Var};
use crate::types::StagedType;

macro_rules! impl_make_fun {
    // Zero parameters
    (0, $make_fun:ident, $make_fun_rec:ident, $FunRef:ident) => {
        impl FunDef {
            pub fn $make_fun<OUT, F, Ret>(
                next_var_id: &mut usize,
                functions: &mut Vec<Option<FunDef>>,
                name: &str,
                body_fn: F,
            ) -> $FunRef<OUT>
            where
                OUT: StagedType,
                F: FnOnce(&mut Ctx) -> Ret,
                Ret: Staged<Out = OUT> + 'static,
            {
                let mut ctx = Ctx::new(*next_var_id);
                let ret = body_fn(&mut ctx);
                *next_var_id = ctx.final_id();

                let func_id = functions.len();
                functions.push(Some(FunDef {
                    name: name.to_string(),
                    body: ctx.into_body(ret),
                    param_infos: vec![],
                    return_info: TypeInfo::from_staged_type::<OUT>(),
                    param_var_ids: vec![],
                }));
                $FunRef::new(func_id)
            }

            pub fn $make_fun_rec<OUT, F, Ret>(
                next_var_id: &mut usize,
                functions: &mut Vec<Option<FunDef>>,
                name: &str,
                body_fn: F,
            ) -> $FunRef<OUT>
            where
                OUT: StagedType,
                F: FnOnce($FunRef<OUT>, &mut Ctx) -> Ret,
                Ret: Staged<Out = OUT> + 'static,
            {
                let func_id = functions.len();
                let func_ref = $FunRef::new(func_id);
                functions.push(None);

                let mut ctx = Ctx::new(*next_var_id);
                let ret = body_fn(func_ref, &mut ctx);
                *next_var_id = ctx.final_id();

                functions[func_id] = Some(FunDef {
                    name: name.to_string(),
                    body: ctx.into_body(ret),
                    param_infos: vec![],
                    return_info: TypeInfo::from_staged_type::<OUT>(),
                    param_var_ids: vec![],
                });
                $FunRef::new(func_id)
            }
        }
    };

    // N parameters (N >= 1)
    ($n:tt, $make_fun:ident, $make_fun_rec:ident, $FunRef:ident, [$($T:ident),+], [$($var:ident),+]) => {
        impl FunDef {
            pub fn $make_fun<$($T,)+ OUT, FN, Ret>(
                next_var_id: &mut usize,
                functions: &mut Vec<Option<FunDef>>,
                name: &str,
                body_fn: FN,
            ) -> $FunRef<$($T,)+ OUT>
            where
                $($T: StagedType,)+
                OUT: StagedType,
                FN: FnOnce(&mut Ctx, $(Var<$T>),+) -> Ret,
                Ret: Staged<Out = OUT> + 'static,
            {
                $(
                    let $var = {
                        let id = *next_var_id;
                        *next_var_id += 1;
                        Var::<$T>::new(id)
                    };
                )+
                let param_var_ids = vec![$($var.id),+];

                let mut ctx = Ctx::new(*next_var_id);
                let ret = body_fn(&mut ctx, $($var),+);
                *next_var_id = ctx.final_id();

                let func_id = functions.len();
                functions.push(Some(FunDef {
                    name: name.to_string(),
                    body: ctx.into_body(ret),
                    param_infos: vec![$(TypeInfo::from_staged_type::<$T>()),+],
                    return_info: TypeInfo::from_staged_type::<OUT>(),
                    param_var_ids,
                }));
                $FunRef::new(func_id)
            }

            pub fn $make_fun_rec<$($T,)+ OUT, FN, Ret>(
                next_var_id: &mut usize,
                functions: &mut Vec<Option<FunDef>>,
                name: &str,
                body_fn: FN,
            ) -> $FunRef<$($T,)+ OUT>
            where
                $($T: StagedType,)+
                OUT: StagedType,
                FN: FnOnce($FunRef<$($T,)+ OUT>, &mut Ctx, $(Var<$T>),+) -> Ret,
                Ret: Staged<Out = OUT> + 'static,
            {
                $(
                    let $var = {
                        let id = *next_var_id;
                        *next_var_id += 1;
                        Var::<$T>::new(id)
                    };
                )+
                let param_var_ids = vec![$($var.id),+];

                let func_id = functions.len();
                let func_ref = $FunRef::new(func_id);
                functions.push(None);

                let mut ctx = Ctx::new(*next_var_id);
                let ret = body_fn(func_ref, &mut ctx, $($var),+);
                *next_var_id = ctx.final_id();

                functions[func_id] = Some(FunDef {
                    name: name.to_string(),
                    body: ctx.into_body(ret),
                    param_infos: vec![$(TypeInfo::from_staged_type::<$T>()),+],
                    return_info: TypeInfo::from_staged_type::<OUT>(),
                    param_var_ids,
                });
                $FunRef::new(func_id)
            }
        }
    };
}

impl_make_fun!(0, make_fun0, make_fun0_rec, FunRef0);
impl_make_fun!(1, make_fun1, make_fun1_rec, FunRef1, [A], [a]);
impl_make_fun!(2, make_fun2, make_fun2_rec, FunRef2, [A, B], [a, b]);
impl_make_fun!(3, make_fun3, make_fun3_rec, FunRef3, [A, B, C], [a, b, c]);
impl_make_fun!(
    4,
    make_fun4,
    make_fun4_rec,
    FunRef4,
    [A, B, C, D],
    [a, b, c, d]
);
impl_make_fun!(
    5,
    make_fun5,
    make_fun5_rec,
    FunRef5,
    [A, B, C, D, E],
    [a, b, c, d, e]
);
impl_make_fun!(
    6,
    make_fun6,
    make_fun6_rec,
    FunRef6,
    [A, B, C, D, E, F],
    [a, b, c, d, e, f]
);
impl_make_fun!(
    7,
    make_fun7,
    make_fun7_rec,
    FunRef7,
    [A, B, C, D, E, F, G],
    [a, b, c, d, e, f, g]
);
impl_make_fun!(
    8,
    make_fun8,
    make_fun8_rec,
    FunRef8,
    [A, B, C, D, E, F, G, H],
    [a, b, c, d, e, f, g, h]
);
