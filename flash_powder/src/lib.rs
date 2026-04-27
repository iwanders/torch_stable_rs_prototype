// What is closest to a torch that works through oxidization? It's magnesium flash powder.

// Strictly follow Rusts' semantics.
// Since it is; String / str
// It will be: Tensor / Ten
//
// We don't put the scalartype in the type, because we can multiply one type to another and have torch handle that.
// We don't put the device type in the type, because that can also be toggled with a global switch easily.

pub mod data;
pub mod methods;
pub mod native_functions;
pub mod tensor;
pub use tensor::{Ten, TenMut, Tensor, TensorAccess};
pub use torch_stable::StableTorchResult;

pub mod prelude {
    use super::*;
    pub use data::{DataAccess, DataManipulationMut};
    pub use methods::TensorMethods;
    pub use native_functions::{NativeFunctions, NativeFunctionsMut};
    pub use tensor::{Ten, TenMut, Tensor, TensorAccess};
}

#[cfg(test)]
mod test {
    use super::*;
    use native_functions::NativeFunctionsOwned;

    #[test]
    fn test_flash_powder_create_error() -> StableTorchResult<()> {
        unsafe {
            torch_stable::stable::c::torch_exception_set_exception_printing(false);
        }
        let a = Tensor::zeros(&[usize::MAX, 5], &Default::default());

        assert!(a.is_err());
        let failure = a.err().unwrap();
        let v = failure.to_string();
        println!("v: {v}");
        assert!(v.contains("(zeros: Dimension size must be non-negative.)"));

        Ok(())
    }
}
