use super::super::super::aoti_torch::*;
use libc::c_char;

// https://github.com/pytorch/pytorch/blob/b5afee0dda86ab028efe66c1eac7028a351edc90/torch/csrc/stable/c/shim.h#L69

// Keep the order the same as the original file.
unsafe extern "C" {
    // https://github.com/pytorch/pytorch/blob/b5afee0dda86ab028efe66c1eac7028a351edc90/torch/csrc/stable/c/shim.h#L69
    pub unsafe fn torch_parse_device_string(
        device_string: *const c_char,
        out_device_type: *mut u32,
        out_device_index: *mut i32,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/7e1205e5321419014c7971a00d9d2292798dfa46/torch/csrc/stable/c/shim.h#L20-L29
    pub unsafe fn torch_call_dispatcher(
        op_name: *const c_char,
        overload_name: *const c_char,
        stack: *const StableIValue,
        extension_build_version: u64,
    ) -> AOTITorchError;
}
