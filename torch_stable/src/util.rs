// I don't really know how to do error handling, since we only ever get a binary signal if it succeeded or not.
// THere's a check for cuda here: https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/macros.h#L9
// The c++ backtrace in the terminal seems hardcoded;
// https://github.com/pytorch/pytorch/blob/c62f72647d6f1bf86025b5e6dbdbd40e6721daf3/torch/csrc/inductor/aoti_torch/utils.h#L14

pub type StableTorchResult<T> = anyhow::Result<T>;

pub struct InhibitLoggingRaii {
    previous_value: bool,
}
impl InhibitLoggingRaii {
    pub fn new() -> Self {
        #[cfg(feature = "use_torch_devel")]
        let previous_value =
            unsafe { crate::stable::c::torch_exception_set_exception_printing(false) };
        #[cfg(not(feature = "use_torch_devel"))]
        let previous_value = false;
        Self { previous_value }
    }
}
impl Drop for InhibitLoggingRaii {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "use_torch_devel")]
            crate::stable::c::torch_exception_set_exception_printing(self.previous_value);
        }
    }
}

#[cfg(feature = "use_torch_devel")]
pub fn get_exception_what() -> String {
    let error_msg_ptr = unsafe { crate::stable::c::torch_exception_get_what_without_backtrace() };

    let error_msg = unsafe { std::ffi::CStr::from_ptr(error_msg_ptr) };
    error_msg
        .to_owned()
        .into_string()
        .unwrap_or("failed to convert error message".to_owned())
}
#[cfg(not(feature = "use_torch_devel"))]
pub fn get_exception_what() -> String {
    return "".to_owned();
}

/// For functions that can fail somewhat gracefully.
#[macro_export]
macro_rules! unsafe_call_bail {
    ($($tokens:tt)*) => {{
        let _ = $crate::util::InhibitLoggingRaii::new();
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            anyhow::bail!("call failed ({}) at {}:{} ({})", code_text, file!(), line!(), $crate::util::get_exception_what());
        }
    }};
}

/// For functions that cannot return a failure like Drop, or functions where it would lead to questionmark soup
/// like getting the dim() on a guaranteed existing tensor pointer.
#[macro_export]
macro_rules! unsafe_call_panic {
    ($($tokens:tt)*) => {{
        let _ = $crate::util::InhibitLoggingRaii::new();
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            panic!("call failed ({}) at {}:{} ({})", code_text, file!(), line!(), $crate::util::get_exception_what());
        }
    }};
}

// This is a macro mostly to ensure we have the correct line number and file :/
#[macro_export]
macro_rules! unsafe_call_dispatch_bail {
    ($op_name:expr, $overload_name:expr, $stack:expr) => {{
        let op_name = std::ffi::CString::new($op_name).expect("CString::new failed");
        let op_name_cstr = op_name.as_ptr();

        let overload_name = std::ffi::CString::new($overload_name).expect("CString::new failed");
        let overload_name_cstr = overload_name.as_ptr();
        let _ = $crate::util::InhibitLoggingRaii::new();

        let api_call_result = unsafe {
            $crate::stable::c::torch_call_dispatcher(
                op_name_cstr,
                overload_name_cstr,
                $stack.as_mut_ptr(),
                $crate::TORCH_ABI_VERSION,
            )
        };
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            anyhow::bail!(
                "dispatch failed ({}, {}) at {}:{} ({})",
                $op_name,
                $overload_name,
                file!(),
                line!(),
                $crate::util::get_exception_what()
            );
        }
    }};
}

#[macro_export]
macro_rules! unsafe_call_dispatch_panic {
    ($op_name:expr, $overload_name:expr, $stack:expr) => {{
        let _ = $crate::util::InhibitLoggingRaii::new();
        let op_name = std::ffi::CString::new($op_name).expect("CString::new failed");
        let op_name_cstr = op_name.as_ptr();

        let overload_name = std::ffi::CString::new($overload_name).expect("CString::new failed");
        let overload_name_cstr = overload_name.as_ptr();

        let api_call_result = unsafe {
            $crate::stable::c::torch_call_dispatcher(
                op_name_cstr,
                overload_name_cstr,
                $stack.as_mut_ptr(),
                $crate::TORCH_ABI_VERSION,
            )
        };
        if api_call_result == $crate::AOTI_TORCH_FAILURE {
            panic!(
                "dispatch failed ({}, {}) at {}:{} ({})",
                $op_name,
                $overload_name,
                file!(),
                line!(),
                $crate::util::get_exception_what()
            );
        }
    }};
}
pub(crate) use unsafe_call_bail;
pub(crate) use unsafe_call_dispatch_bail;
pub(crate) use unsafe_call_dispatch_panic;
pub(crate) use unsafe_call_panic;
