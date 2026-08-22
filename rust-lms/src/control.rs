//! Control flow constructs for staged computations.
//!
//! This module provides:
//! - `IfThenElse<COND, THEN, ELSE>`: Conditional expression with both branches
//! - `IfThen<COND, BODY>`: Conditional for side effects (returns unit)
//! - `While<COND, BODY>`: While loop (returns unit)
//!
//! Note: For sequencing, use tuples instead (see `tuple.rs`).

use cranelift_codegen::ir::Value;

use crate::staged::{CompilationContext, IntoStaged, Staged};
use crate::types::StagedType;

// =============================================================================
// IfThenElse<COND, THEN, ELSE> - Conditional Expression
// =============================================================================

/// Conditional expression: if condition then return THEN else return ELSE.
///
/// Both branches must produce the same type.
///
/// # Example
/// ```ignore
/// // Clamp x to maximum 10
/// let clamped = if_then_else(lt(x, Const::new(10)), x, Const::new(10));
/// ```
#[derive(Clone)]
pub struct IfThenElse<COND, THEN, ELSE> {
    condition: COND,
    then_branch: THEN,
    else_branch: ELSE,
}

unsafe impl<COND, THEN, ELSE, T> Staged for IfThenElse<COND, THEN, ELSE>
where
    COND: Staged<Out = bool>,
    THEN: Staged<Out = T>,
    ELSE: Staged<Out = T>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate condition code in current block
        let cond_val = self.condition.codegen(ctx);

        // Create the three blocks we need
        let then_block = ctx.create_block();
        let else_block = ctx.create_block();
        let merge_block = ctx.create_block();

        // Add block parameter to merge_block to receive the result (phi node)
        ctx.append_block_param(merge_block, T::scalar_type());

        // Branch based on condition
        ctx.brif(cond_val, then_block, &[], else_block, &[]);

        // Generate then branch
        ctx.switch_to_block(then_block);
        ctx.seal_block(then_block); // Single predecessor (entry block)
        let then_val = self.then_branch.codegen(ctx);
        ctx.jump(merge_block, &[then_val]);

        // Generate else branch
        ctx.switch_to_block(else_block);
        ctx.seal_block(else_block); // Single predecessor (entry block)
        let else_val = self.else_branch.codegen(ctx);
        ctx.jump(merge_block, &[else_val]);

        // Continue in merge block
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block); // Two predecessors now known

        // Return the block parameter (the merged value)
        ctx.block_param(merge_block, 0)
    }
}

/// Create a conditional expression
///
/// Accepts any value that can be converted into a bool staged expression for the condition.
/// This allows ergonomic usage like `if_then_else(true, x, y)` instead of
/// `if_then_else(Const::<bool>::new(true), x, y)`.
pub fn if_then_else<C, THEN, ELSE, T>(
    condition: C,
    then_branch: THEN,
    else_branch: ELSE,
) -> IfThenElse<C::Staged, THEN, ELSE>
where
    C: IntoStaged<bool>,
    THEN: Staged<Out = T>,
    ELSE: Staged<Out = T>,
    T: StagedType,
{
    IfThenElse {
        condition: condition.into_staged(),
        then_branch,
        else_branch,
    }
}

// =============================================================================
// IfThen<COND, BODY> - Conditional for Side Effects
// =============================================================================

/// Conditional for side effects: if condition then execute body, otherwise skip.
///
/// Returns `()` regardless of whether body was executed.
/// Body must also produce `()` (use for side-effect-only operations).
///
/// # Example
/// ```ignore
/// // Clamp x to minimum 0
/// if_then(lt(x, Const::new(0)), assign(x, Const::new(0)));
/// ```
#[derive(Clone)]
pub struct IfThen<COND, BODY> {
    condition: COND,
    body: BODY,
}

unsafe impl<COND, BODY> Staged for IfThen<COND, BODY>
where
    COND: Staged<Out = bool>,
    BODY: Staged<Out = ()>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate condition code in current block
        let cond_val = self.condition.codegen(ctx);

        // Create the two blocks we need
        let then_block = ctx.create_block();
        let merge_block = ctx.create_block();

        // Branch: if true go to then_block, else skip to merge_block
        ctx.brif(cond_val, then_block, &[], merge_block, &[]);

        // Generate then branch (body)
        ctx.switch_to_block(then_block);
        ctx.seal_block(then_block); // Single predecessor
        let _ = self.body.codegen(ctx); // Execute for side effects
        ctx.jump(merge_block, &[]);

        // Continue in merge block
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block); // Two predecessors now known

        // Return unit value
        ctx.get_unit_value()
    }
}

/// Create a conditional for side effects
///
/// Accepts any value that can be converted into a bool staged expression for the condition.
pub fn if_then<C, BODY>(condition: C, body: BODY) -> IfThen<C::Staged, BODY>
where
    C: IntoStaged<bool>,
    BODY: Staged<Out = ()>,
{
    IfThen {
        condition: condition.into_staged(),
        body,
    }
}
// =============================================================================
// While<COND, BODY> - While Loop
// =============================================================================

/// While loop: execute body while condition is true.
///
/// Returns `()`. Both condition and body are re-evaluated each iteration.
/// The condition is checked at the start of each iteration (pre-check loop).
///
/// # Example
/// ```ignore
/// // Compute sum of 1..=n (stored in 'sum' variable)
/// // i = 1; sum = 0;
/// // while (i <= n) { sum = sum + i; i = i + 1; }
/// let i = compiler.var::<i64>();
/// let sum = compiler.var::<i64>();
/// seq(
///     seq(assign(i, Const::new(1)), assign(sum, Const::new(0))),
///     seq(
///         while_loop(
///             lt(i, add(n, Const::new(1))),  // i <= n
///             seq(
///                 assign(sum, add(sum, i)),
///                 assign(i, add(i, Const::new(1))),
///             ),
///         ),
///         sum,
///     ),
/// )
/// ```
#[derive(Clone)]
pub struct While<COND, BODY> {
    condition: COND,
    body: BODY,
}

unsafe impl<COND, BODY> Staged for While<COND, BODY>
where
    COND: Staged<Out = bool>,
    BODY: Staged<Out = ()>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Create the blocks for the loop structure
        let loop_header = ctx.create_block();
        let loop_body = ctx.create_block();
        let loop_exit = ctx.create_block();

        // Jump from current block to loop header
        ctx.jump(loop_header, &[]);

        // Loop header: evaluate condition and branch
        ctx.switch_to_block(loop_header);
        // DON'T seal loop_header yet - it has two predecessors (entry and loop_body)
        // We'll seal it after generating the back-edge from loop_body

        let cond_val = self.condition.codegen(ctx);
        ctx.brif(cond_val, loop_body, &[], loop_exit, &[]);

        // Loop body: execute body and jump back to header
        ctx.switch_to_block(loop_body);
        ctx.seal_block(loop_body); // Single predecessor (loop_header)
        let _ = self.body.codegen(ctx);
        ctx.jump(loop_header, &[]);

        // Now we can seal loop_header - both predecessors are known
        ctx.seal_block(loop_header);

        // Loop exit
        ctx.switch_to_block(loop_exit);
        ctx.seal_block(loop_exit); // Single predecessor (loop_header)

        // Return unit value
        ctx.get_unit_value()
    }
}

/// Create a while loop
///
/// Accepts any value that can be converted into a bool staged expression for the condition.
/// This allows ergonomic usage like `while_loop(true, body)` instead of
/// `while_loop(Const::<bool>::new(true), body)`.
pub fn while_loop<C, BODY>(condition: C, body: BODY) -> While<C::Staged, BODY>
where
    C: IntoStaged<bool>,
    BODY: Staged<Out = ()>,
{
    While {
        condition: condition.into_staged(),
        body,
    }
}

// =============================================================================
// Not<C> - Logical negation of a boolean
// =============================================================================

/// Logical negation: `!cond`.
#[derive(Clone)]
pub struct Not<C> {
    cond: C,
}

unsafe impl<C> Staged for Not<C>
where
    C: Staged<Out = bool>,
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let v = self.cond.codegen(ctx);
        // bool is an i8 in {0, 1}; `v == 0` is its negation.
        ctx.icmp_imm(cranelift_codegen::ir::condcodes::IntCC::Equal, v, 0)
    }
}

/// Negate a boolean staged expression: `not(cond)`.
pub fn not<C>(cond: C) -> Not<C::Staged>
where
    C: IntoStaged<bool>,
{
    Not {
        cond: cond.into_staged(),
    }
}
