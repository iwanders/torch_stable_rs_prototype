// I don't really know how to do error handling, since we only ever get a binary signal if it succeeded or not.
// THere's a check for cuda here: https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/macros.h#L9
// The c++ backtrace in the terminal seems hardcoded;
// https://github.com/pytorch/pytorch/blob/c62f72647d6f1bf86025b5e6dbdbd40e6721daf3/torch/csrc/inductor/aoti_torch/utils.h#L14

pub type StableTorchResult<T> = anyhow::Result<T>;

macro_rules! unsafe_call_bail {
    ($($tokens:tt)*) => {{
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == crate::AOTI_TORCH_FAILURE {
            bail!("call failed ({}) at {}:{}", code_text, file!(), line!());
        }
    }};
}

pub(crate) use unsafe_call_bail;
