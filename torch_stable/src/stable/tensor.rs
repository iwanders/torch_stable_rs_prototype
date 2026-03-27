// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_inl.h

use super::device::DeviceIndex;
use crate::aoti_torch::{AtenTensorHandle, aoti_torch_delete_tensor_object};
use crate::headeronly::core::{Layout, ScalarType};

use crate::aoti_torch::*;
use crate::stable::device::{Device, DeviceType};

use std::sync::Arc;

use crate::{StableTorchResult, unsafe_call_bail, unsafe_call_panic};
use anyhow::{anyhow, bail};

struct Tensordropper(AtenTensorHandle);
impl Drop for Tensordropper {
    fn drop(&mut self) {
        // We can't do anything with the return value here, so we quietly ignore it :/
        unsafe_call_panic!(aoti_torch_delete_tensor_object(self.0));
    }
}

#[derive(Clone)]
pub struct Tensor {
    ath: Arc<Tensordropper>,
}
impl Tensor {
    /// Creates a new uninitialised tensor
    /// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h#L55
    pub fn new() -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_new_uninitialized_tensor(&mut handle_res));
        Ok(Self {
            ath: Arc::new(Tensordropper(handle_res)),
        })
    }

    /// Direct access to the tensor handle.
    pub fn get(&self) -> AtenTensorHandle {
        (*self.ath).0
    }

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h#L126
    // Do we need all these pointers? Lets try without it for now.

    // Tensor itself has this as i64 :/
    /// Returns the number of dimensions of the tensor.
    /// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h#L204
    pub fn dim(&self) -> usize {
        let mut dim: i64 = 0;
        unsafe_call_panic!(aoti_torch_get_dim(self.get(), &mut dim));
        return dim as usize;
    }

    pub fn numel(&self) -> usize {
        let mut numel: i64 = 0;
        unsafe_call_panic!(aoti_torch_get_numel(self.get(), &mut numel));
        return numel as usize;
    }

    pub fn sizes(&self) -> &[usize] {
        // Oof, yuck, casting i64 to usize with reinterpret cast... but the alternative is using i64 for indexing.
        let mut i64_size_ptr: *const i64 = std::ptr::null();
        unsafe_call_panic!(aoti_torch_get_sizes(self.get(), &mut i64_size_ptr));
        let len = self.dim();
        let usize_ptr = i64_size_ptr.cast::<usize>();
        return unsafe { std::slice::from_raw_parts(usize_ptr, len) };
    }

    pub fn strides(&self) -> &[usize] {
        // Oof, yuck, casting i64 to usize with reinterpret cast... but the alternative is using i64 for indexing.
        let mut i64_size_ptr: *const i64 = std::ptr::null();
        unsafe_call_panic!(aoti_torch_get_strides(self.get(), &mut i64_size_ptr));
        let len = self.dim();
        let usize_ptr = i64_size_ptr.cast::<usize>();
        return unsafe { std::slice::from_raw_parts(usize_ptr, len) };
    }

    pub fn is_contiguous(&self) -> bool {
        let mut is_contiguous: bool = true;
        unsafe_call_panic!(aoti_torch_is_contiguous(self.get(), &mut is_contiguous));
        return is_contiguous;
    }

    pub fn stride(&self, dim: usize) -> usize {
        let mut stride: i64 = 0;
        unsafe_call_panic!(aoti_torch_get_stride(self.get(), dim as i64, &mut stride));
        return stride as usize;
    }

    pub fn get_device_index(&self) -> DeviceIndex {
        let mut device_index: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_device_index(self.get(), &mut device_index));
        return DeviceIndex(device_index);
    }

    pub fn is_cuda(&self) -> bool {
        let mut device_type: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_device_type(self.get(), &mut device_type));
        device_type == DeviceType::CUDA as i32
    }

    pub fn is_cpu(&self) -> bool {
        let mut device_type: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_device_type(self.get(), &mut device_type));
        device_type == DeviceType::CPU as i32
    }

    pub fn size(&self, dim: usize) -> usize {
        let mut size: i64 = 0;
        unsafe_call_panic!(aoti_torch_get_size(self.get(), dim as i64, &mut size));
        return size as usize;
    }

    // Function name should've been is_defined, but lets go with the following the upstream naming.
    pub fn defined(&self) -> bool {
        let mut is_defined: bool = false;
        unsafe_call_panic!(aoti_torch_is_defined(self.get(), &mut is_defined));
        return is_defined;
    }

    // storage_offset, element_size

    pub fn scalar_type(&self) -> ScalarType {
        let mut dtype: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_dtype(self.get(), &mut dtype));
        return ScalarType::try_from(dtype).unwrap();
    }

    pub fn device(&self) -> Device {
        let mut device_index: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_device_index(self.get(), &mut device_index));
        let device_index = DeviceIndex(device_index);
        let mut device_type: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_device_type(self.get(), &mut device_type));
        let device_type = DeviceType::try_from(device_type as u32).unwrap();

        Device::from_parts(device_type, device_index)
    }
    pub fn layout(&self) -> Layout {
        let mut layout: i32 = 0;
        unsafe_call_panic!(aoti_torch_get_layout(self.get(), &mut layout));
        return Layout::try_from(layout).unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_tensor_uninitialised() {
        // Mostly to check if valgrind is clean with this test, to see if the dropping mechanism works, and it does.
        let t = Tensor::new().expect("should succeed");
        assert_eq!(t.numel(), 0);
        assert_eq!(t.sizes(), &[0]);
        //assert_eq!(t.strides(), &[0]); // Calling on uninitialised tensor is an error.
        assert_eq!(t.is_contiguous(), true); // Calling on uninitialised tensor is an error.
        // assert_eq!(t.stride(0), 0); // Calling on uninitialised tensor is an error.
        // assert_eq!(t.get_device(), DeviceIndex(-1));
        assert_eq!(t.defined(), false);
        assert_eq!(t.scalar_type(), ScalarType::Undefined);
        // assert_eq!(t.device(), Device::from_str("cpu").unwrap());
    }
}
