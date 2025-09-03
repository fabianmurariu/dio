//! # Dio v4: Staged Compilation Framework
//!
//! This crate implements the next-generation Dio query compiler using staging techniques
//! and callback-based operator fusion as described in `docs/dio4.md`.

pub mod staging;
pub mod filter;
pub mod compiler;