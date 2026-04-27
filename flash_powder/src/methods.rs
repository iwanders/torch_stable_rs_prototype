use crate::{Ten, TenMut, Tensor, TensorAccess};
use torch_stable::headeronly::core::ScalarType;
use torch_stable::stable::device::Device;
pub trait TensorMethods: TensorAccess {
    fn dim(&self) -> usize {
        self.get_tensor().dim()
    }

    fn scalar_type(&self) -> ScalarType {
        self.get_tensor().scalar_type()
    }

    fn element_size(&self) -> usize {
        self.get_tensor().element_size()
    }

    fn device(&self) -> Device {
        self.get_tensor().device()
    }

    fn storage_offset(&self) -> usize {
        self.get_tensor().storage_offset()
    }

    fn numel(&self) -> usize {
        self.get_tensor().numel()
    }

    fn is_cpu(&self) -> bool {
        self.get_tensor().is_cpu()
    }

    fn const_data_ptr(&self) -> *const u8 {
        self.get_tensor().const_data_ptr()
    }

    fn mutable_data_ptr(&self) -> *mut u8 {
        self.get_tensor().mutable_data_ptr()
    }
}
impl TensorMethods for Tensor {}
impl<'a> TensorMethods for Ten<'a> {}
impl<'a> TensorMethods for TenMut<'a> {}
