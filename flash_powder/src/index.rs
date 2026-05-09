//! Indexing

// todo!!
use crate::properties::TensorProperties;
use crate::{StableTorchResult, Tensor, TensorAccess};
use torch_stable::aoti_torch::{AtenTensorHandle, aoti_torch_zero_};
use torch_stable::headeronly::core::{Layout, ScalarType};
use torch_stable::stable::device::Device;
pub use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_bail,
    unsafe_call_dispatch_bail,
};

pub trait TensorIndex: TensorAccess + TensorProperties {}
