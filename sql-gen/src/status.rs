//! Query-owned runtime error carrier for JIT callbacks.
//!
//! `DataFusionError` is not FFI-safe and must not cross the JIT ABI. A fallible
//! `extern "C"` callback therefore records the detailed error here (Rust-owned)
//! and returns only an ABI-safe sentinel to generated code:
//!
//! - pointer-returning callbacks return a **null pointer**, where null is not a
//!   valid result (e.g. the `group_upsert*` record-pointer family);
//! - unit-returning callbacks return a [`CallbackStatus`] (lowered to `bool` at the
//!   ABI — see its docs).
//!
//! Generated code branches to its error epilogue on the sentinel; after the kernel
//! returns, the driver calls [`RuntimeStatus::check`] to recover the `Result`.
//!
//! This generalizes the `Inputs::error` pattern already used by the scan stream so
//! grouping, joins, and output building can share one carrier.

use datafusion_common::{DataFusionError, Result};

/// First-error-wins error sink embedded in a query's runtime state
/// (`GroupState`, `JoinState`, the output builder, …).
#[derive(Default)]
pub struct RuntimeStatus {
    error: Option<DataFusionError>,
}

impl RuntimeStatus {
    /// Record the first error. Later errors are usually consequences of it, so the
    /// original is the one worth surfacing.
    pub fn record_error(&mut self, error: DataFusionError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// Whether an error has been recorded. Once poisoned, a callback should return
    /// its sentinel and stop mutating partially-valid state rather than continue.
    pub fn is_failed(&self) -> bool {
        self.error.is_some()
    }

    /// Take the recorded error, converting the run into a `Result`. Called by the
    /// driver after the kernel returns.
    pub fn check(&mut self) -> Result<()> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// ABI-safe status returned by fallible **unit** callbacks — those with no free
/// pointer/`u64` sentinel in their result (`join_insert`, the `group_key_*`
/// pushers, `strview_append*`). Generated code branches to its error epilogue on
/// [`CallbackStatus::Failed`].
///
/// Although declared `#[repr(u8)]`, this lowers to **`bool`** where it crosses the
/// JIT ABI (`true` == `Failed`): rust-lms models a 1-byte status as `bool`, and a
/// staged `u8` enum could otherwise carry an invalid discriminant. Use
/// [`CallbackStatus::failed`]/[`CallbackStatus::from_failed`] at the `extern "C"`
/// boundary.
// First used in the join/strview increment (fallible unit callbacks); defined now
// so the two-mechanism design lives in one place.
#[allow(dead_code)]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallbackStatus {
    Ok = 0,
    Failed = 1,
}

#[allow(dead_code)]
impl CallbackStatus {
    /// The ABI representation: `true` when the call failed.
    pub fn failed(self) -> bool {
        self == CallbackStatus::Failed
    }

    /// Reconstruct from the ABI boolean.
    pub fn from_failed(failed: bool) -> Self {
        if failed {
            CallbackStatus::Failed
        } else {
            CallbackStatus::Ok
        }
    }
}
