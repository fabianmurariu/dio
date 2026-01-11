//! Control flow constructs for staged computations.
//!
//! This module provides:
//! - `Seq<A, B>`: Execute A for side effects, then return B's value
//! - `IfThenElse<COND, THEN, ELSE>`: Conditional expression with both branches
//! - `IfThen<COND, BODY>`: Conditional for side effects (returns unit)
//! - `While<COND, BODY>`: While loop (returns unit)

use cranelift_codegen::ir::{types, BlockArg, InstBuilder, Value};

use crate::staged::{CompilationContext, Staged};
use crate::types::{BoolType, StagedType, UnitType};

// =============================================================================
// Seq<A, B> - Statement Sequencing
// =============================================================================

/// Sequence two expressions: execute first for side effects, return second's value.
///
/// # Example
/// ```ignore
/// let x = compiler.var::<I64Type>();
/// let expr = seq(assign(x, Const::new(5)), x);  // Assigns 5 to x, then returns x
/// ```
#[derive(Clone)]
pub struct Seq<A, B> {
    first: A,
    second: B,
}

impl<A, B, T> Staged for Seq<A, B>
where
    A: Staged,           // First can produce any type (executed for side effects)
    B: Staged<Out = T>,  // Second determines the result type
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Execute first expression, discard its value
        let _ = self.first.codegen(ctx);
        // Execute second expression and return its value
        self.second.codegen(ctx)
    }
}

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

impl<COND, THEN, ELSE, T> Staged for IfThenElse<COND, THEN, ELSE>
where
    COND: Staged<Out = BoolType>,
    THEN: Staged<Out = T>,
    ELSE: Staged<Out = T>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate condition code in current block
        let cond_val = self.condition.codegen(ctx);

        // Create the three blocks we need
        let then_block = ctx.builder.create_block();
        let else_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();

        // Add block parameter to merge_block to receive the result (phi node)
        let result_type = T::cranelift_type();
        ctx.builder.append_block_param(merge_block, result_type);

        // Branch based on condition
        ctx.builder.ins().brif(cond_val, then_block, &[], else_block, &[]);

        // Generate then branch
        ctx.builder.switch_to_block(then_block);
        ctx.builder.seal_block(then_block); // Single predecessor (entry block)
        let then_val = self.then_branch.codegen(ctx);
        ctx.builder.ins().jump(merge_block, &[BlockArg::Value(then_val)]);

        // Generate else branch
        ctx.builder.switch_to_block(else_block);
        ctx.builder.seal_block(else_block); // Single predecessor (entry block)
        let else_val = self.else_branch.codegen(ctx);
        ctx.builder.ins().jump(merge_block, &[BlockArg::Value(else_val)]);

        // Continue in merge block
        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block); // Two predecessors now known

        // Return the block parameter (the merged value)
        ctx.builder.block_params(merge_block)[0]
    }
}

/// Create a conditional expression
pub fn if_then_else<COND, THEN, ELSE, T>(
    condition: COND,
    then_branch: THEN,
    else_branch: ELSE,
) -> IfThenElse<COND, THEN, ELSE>
where
    COND: Staged<Out = BoolType>,
    THEN: Staged<Out = T>,
    ELSE: Staged<Out = T>,
    T: StagedType,
{
    IfThenElse {
        condition,
        then_branch,
        else_branch,
    }
}

// =============================================================================
// IfThen<COND, BODY> - Conditional for Side Effects
// =============================================================================

/// Conditional for side effects: if condition then execute body, otherwise skip.
///
/// Returns `UnitType` regardless of whether body was executed.
/// Body must also produce `UnitType` (use for side-effect-only operations).
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

impl<COND, BODY> Staged for IfThen<COND, BODY>
where
    COND: Staged<Out = BoolType>,
    BODY: Staged<Out = UnitType>,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate condition code in current block
        let cond_val = self.condition.codegen(ctx);

        // Create the two blocks we need
        let then_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();

        // Branch: if true go to then_block, else skip to merge_block
        ctx.builder.ins().brif(cond_val, then_block, &[], merge_block, &[]);

        // Generate then branch (body)
        ctx.builder.switch_to_block(then_block);
        ctx.builder.seal_block(then_block); // Single predecessor
        let _ = self.body.codegen(ctx); // Execute for side effects
        ctx.builder.ins().jump(merge_block, &[]);

        // Continue in merge block
        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block); // Two predecessors now known

        // Return unit value
        ctx.builder.ins().iconst(types::I8, 0)
    }
}

/// Create a conditional for side effects
pub fn if_then<COND, BODY>(condition: COND, body: BODY) -> IfThen<COND, BODY>
where
    COND: Staged<Out = BoolType>,
    BODY: Staged<Out = UnitType>,
{
    IfThen { condition, body }
}
// =============================================================================
// While<COND, BODY> - While Loop
// =============================================================================

/// While loop: execute body while condition is true.
///
/// Returns `UnitType`. Both condition and body are re-evaluated each iteration.
/// The condition is checked at the start of each iteration (pre-check loop).
///
/// # Example
/// ```ignore
/// // Compute sum of 1..=n (stored in 'sum' variable)
/// // i = 1; sum = 0;
/// // while (i <= n) { sum = sum + i; i = i + 1; }
/// let i = compiler.var::<I64Type>();
/// let sum = compiler.var::<I64Type>();
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

impl<COND, BODY> Staged for While<COND, BODY>
where
    COND: Staged<Out = BoolType>,
    BODY: Staged<Out = UnitType>,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Create the blocks for the loop structure
        let loop_header = ctx.builder.create_block();
        let loop_body = ctx.builder.create_block();
        let loop_exit = ctx.builder.create_block();

        // Jump from current block to loop header
        ctx.builder.ins().jump(loop_header, &[]);

        // Loop header: evaluate condition and branch
        ctx.builder.switch_to_block(loop_header);
        // DON'T seal loop_header yet - it has two predecessors (entry and loop_body)
        // We'll seal it after generating the back-edge from loop_body

        let cond_val = self.condition.codegen(ctx);
        ctx.builder.ins().brif(cond_val, loop_body, &[], loop_exit, &[]);

        // Loop body: execute body and jump back to header
        ctx.builder.switch_to_block(loop_body);
        ctx.builder.seal_block(loop_body); // Single predecessor (loop_header)
        let _ = self.body.codegen(ctx);
        ctx.builder.ins().jump(loop_header, &[]);

        // Now we can seal loop_header - both predecessors are known
        ctx.builder.seal_block(loop_header);

        // Loop exit
        ctx.builder.switch_to_block(loop_exit);
        ctx.builder.seal_block(loop_exit); // Single predecessor (loop_header)

        // Return unit value
        ctx.builder.ins().iconst(types::I8, 0)
    }
}

/// Create a while loop
pub fn while_loop<COND, BODY>(condition: COND, body: BODY) -> While<COND, BODY>
where
    COND: Staged<Out = BoolType>,
    BODY: Staged<Out = UnitType>,
{
    While { condition, body }
}
