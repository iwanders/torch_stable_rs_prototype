use crate::{Ten, TenMut, Tensor, TensorAccess};
use torch_stable::headeronly::core::{Layout, ScalarType};
use torch_stable::stable::device::{Device, DeviceIndex};
pub trait TensorMethods: TensorAccess {
    fn dim(&self) -> usize {
        self.get_tensor().dim()
    }

    fn numel(&self) -> usize {
        self.get_tensor().numel()
    }

    fn sizes(&self) -> &[usize] {
        self.get_tensor().sizes()
    }

    fn strides(&self) -> &[usize] {
        self.get_tensor().strides()
    }

    fn stride(&self, dim: usize) -> usize {
        self.get_tensor().stride(dim)
    }

    fn is_contiguous(&self) -> bool {
        self.get_tensor().is_contiguous()
    }

    fn scalar_type(&self) -> ScalarType {
        self.get_tensor().scalar_type()
    }
    fn layout(&self) -> Layout {
        self.get_tensor().layout()
    }

    fn device(&self) -> Device {
        self.get_tensor().device()
    }

    fn device_index(&self) -> DeviceIndex {
        self.get_tensor().get_device_index()
    }

    fn is_cpu(&self) -> bool {
        self.get_tensor().is_cpu()
    }

    fn is_cuda(&self) -> bool {
        self.get_tensor().is_cpu()
    }

    fn size(&self, dim: usize) -> usize {
        self.get_tensor().size(dim)
    }
    fn is_defined(&self) -> bool {
        self.get_tensor().defined()
    }
    fn element_size(&self) -> usize {
        self.get_tensor().element_size()
    }

    fn storage_offset(&self) -> usize {
        self.get_tensor().storage_offset()
    }

    fn data_ptr(&self) -> *const u8 {
        self.get_tensor().data_ptr()
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
