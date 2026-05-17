//! This holds the C shims from stable.
//!
//! Since C api's don't have namespaces, neither does this module to ensure things are robust against upstream moving
//! function signatures around between the files.
mod shim;
pub use shim::*;
