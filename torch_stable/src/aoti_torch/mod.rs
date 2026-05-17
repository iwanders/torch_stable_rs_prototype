//! This holds the C shims from aoti_torch.
//!
//! Since C api's don't have namespaces, neither does this module to ensure things are robust against upstream moving
//! function signatures around between the files.
mod c;
mod generated;
pub use c::*;
pub use generated::*;
