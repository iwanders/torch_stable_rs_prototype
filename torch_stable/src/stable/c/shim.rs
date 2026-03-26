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
}
