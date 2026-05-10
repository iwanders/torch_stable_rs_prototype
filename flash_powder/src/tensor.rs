//! Holds the three Tensor types.
use crate::StableTorchResult;
use crate::core_methods::CoreMethods;
use crate::data::DataRef;
use crate::properties::TensorProperties;
use anyhow;
use torch_stable::unsafe_call_dispatch_panic;
use torch_stable::{aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor};

/// A tensor, this owns its data.
///
/// Interact with it through any of the traits that are implemented for [`TensorAccess`].
///
/// Usually you don't create this directly, but create tensors through [`crate::factory::TensorFactory`].
pub struct Tensor {
    tensor: StableTensor,
}
impl Clone for Tensor {
    /// This is a full owning clone, but lazy.
    ///
    /// Under the hood this calls <https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1278>, so the `_lazy_clone` kernel.
    ///
    /// The docs for that function state:
    ///
    /// > Like clone, but the copy takes place lazily, only if either the input or the output are written.
    ///
    /// Since clone can't fail in rust, I chose this because a lazy clone is unlikely to cause an out of memory error.
    ///
    /// It does mean that memory allocation errors are deffered to later in the program, but hopefully they can be handled there.
    fn clone(&self) -> Self {
        // Clone cannot throw... so we use a lazy clone; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1278
        let mut stack: [StableIValue; 1] = [(&self.tensor).into()];
        unsafe_call_dispatch_panic!("aten::_lazy_clone", "", stack.as_mut_slice());
        let r: Tensor = Self::new(stack[0].try_into().unwrap());

        r
    }
}

impl Tensor {
    /// Create a new tensor backed by the provided StableTensor.
    ///
    /// The provided tensor should be detached from anything else and exclusive ownership should be passed.
    pub fn new(tensor: StableTensor) -> Self {
        Self { tensor }
    }

    /// Equivalent to torch.tensor(data)
    ///
    /// Always allocates in the provided data type, on the cpu.
    ///
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.tensor.html#torch.tensor>)
    ///
    /// Is actually implemented via TryInto
    pub fn from<T>(data: T) -> StableTorchResult<Tensor>
    where
        T: TryInto<Tensor>,
        T::Error: Into<anyhow::Error>,
    {
        let b: StableTorchResult<Tensor> = data.try_into().map_err(|e| e.into());
        b
    }
}

/// A borrow on another Tensor, like a view into one.
pub struct Ten<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    _parent: &'a StableTensor,
}
impl<'a> Ten<'a> {
    pub fn new(parent: &'a StableTensor, tensor: StableTensor) -> Self {
        Self {
            _parent: parent,
            tensor,
        }
    }
    pub(crate) fn as_parent(&self) -> &'a StableTensor {
        self._parent
    }
}

/// A mutable borrow on another Tensor, like mutably borrowed slice into one.
pub struct TenMut<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    _parent: &'a mut StableTensor,
}
impl<'a> TenMut<'a> {
    pub fn new(parent: &'a mut StableTensor, tensor: StableTensor) -> Self {
        Self {
            _parent: parent,
            tensor,
        }
    }
    pub(crate) fn into_parent(self) -> &'a mut StableTensor {
        self._parent
    }
}

pub trait TensorAccess {
    fn get_tensor(&self) -> &StableTensor;
    fn get_tensor_mut(&mut self) -> &mut StableTensor;
}

impl<'a> TensorAccess for TenMut<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl<'a> TensorAccess for Ten<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

impl TensorAccess for Tensor {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
    fn get_tensor_mut(&mut self) -> &mut StableTensor {
        &mut self.tensor
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PrintOptions {
    pub precision: usize,
    pub threshold: usize,
    pub edgeitems: usize,
    pub linewidth: usize,
}
impl PrintOptions {
    pub const fn new() -> Self {
        Self {
            precision: 4,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 80,
        }
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            precision: 4,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 80,
        }
    }
}
// And a local helper because https://github.com/rust-lang/rust/issues/118117 is not stabilised yet.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarPrintOptions {
    pub precision: Option<usize>,
    pub width: Option<usize>,
}
impl ScalarPrintOptions {
    pub fn format<T: std::fmt::Debug>(&self, value: &T) -> String {
        match (self.precision, self.width) {
            (None, None) => format!("{:?}", value),
            (None, Some(width)) => format!("{: >width$?}", value),
            (Some(precision), None) => format!("{:.precision$?}", value),
            (Some(precision), Some(width)) => format!("{: >width$.precision$?}", value),
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub struct TensorPrintOptions {
    pub print_options: PrintOptions,
    pub scalar_options: ScalarPrintOptions,
    pub prefix: &'static str,
}

// formatter's new method is experimental... so I can't create a formatter to just test my printoptions.
use std::sync::Mutex;
static GLOBAL_PRINT_YUCK: Mutex<PrintOptions> = Mutex::new(PrintOptions::new());

/// Set the options for printing.
///
/// [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.set_printoptions.html#torch.set_printoptions)
///
/// This writes into a global singleton that is NOT thread_local.
pub fn set_printoptions(options: &PrintOptions) {
    *(GLOBAL_PRINT_YUCK.lock().unwrap()) = *options;
}

// https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L130
//
// They do two passess, one to determine the width of the elements, then another to actually print.
pub fn tensor_format<T: TensorAccess + TensorProperties + CoreMethods + DataRef>(
    t: &T,
    options: &TensorPrintOptions,
) -> String {
    // https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L242
    let vector_str = |indent: usize, o: &T, element_width: usize, summarize: bool| -> String {
        let mut r = String::new();
        let element_length = element_width + ", ".len();
        let elements_per_line =
            ((options.print_options.linewidth - indent) / element_length).max(1);
        let mut local_options = options.scalar_options;
        local_options.width = Some(element_width);

        let mut element_str = vec![];
        if summarize {
            let left = o.narrow(0, 0, options.print_options.edgeitems).unwrap();
            element_str
                .extend((0..left.numel()).map(|i| left.d_fmt(&[i], &local_options).unwrap()));
            element_str.push("...".to_owned());
            let right = o
                .narrow(
                    0,
                    -(options.print_options.edgeitems as isize),
                    options.print_options.edgeitems,
                )
                .unwrap();
            element_str
                .extend((0..right.numel()).map(|i| right.d_fmt(&[i], &local_options).unwrap()));
        } else {
            element_str = (0..o.numel())
                .map(|i| o.d_fmt(&[i], &local_options).unwrap())
                .collect();
        }
        let lines = element_str.chunks(elements_per_line);
        let lines: Vec<String> = lines.map(|z| z.join(", ")).collect();
        r += "[";
        let joiner = format!(",\n{}", " ".repeat(indent));
        r += &lines.join(&joiner);
        r += "]";
        r
    };
    let determine_width = |o: &T| -> usize {
        let mut m = 0;
        let linear = o.view(&[o.numel()]).unwrap();
        for i in 0..o.numel() {
            m = m.max(linear.d_fmt(&[i], &options.scalar_options).unwrap().len())
        }
        m
    };

    let summarize = t.numel() > options.print_options.threshold;
    if t.dim() == 0 {
        format!(
            "{}({})",
            options.prefix,
            t.d_fmt(&[], &options.scalar_options).unwrap()
        )
    } else if t.dim() == 1 {
        let element_width = determine_width(t);

        let mut r = format!("{}(", options.prefix);
        r += &vector_str(options.prefix.len() + 2, t, element_width, summarize);
        r += ")\n";
        r
    } else {
        todo!()
    }
}
fn tensor_format_with_formatter<T: TensorAccess + TensorProperties + CoreMethods + DataRef>(
    t: &T,
    prefix: &'static str,
    fmt: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let print_options = {
        let l = GLOBAL_PRINT_YUCK.lock().unwrap();
        *l
    };

    let mut scalar_options = ScalarPrintOptions::default();
    if let Some(provided_precision) = fmt.precision() {
        scalar_options.precision = Some(provided_precision);
    } else {
        scalar_options.precision = Some(print_options.precision);
    }

    let print_options = TensorPrintOptions {
        print_options,
        scalar_options,
        prefix,
    };

    let tensor_formatted = tensor_format(t, &print_options);
    fmt.write_fmt(format_args!("{}", tensor_formatted))
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        tensor_format_with_formatter(self, "Tensor", f)
    }
}

impl std::fmt::Debug for TenMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        tensor_format_with_formatter(self, "TenMut", f)
    }
}

impl std::fmt::Debug for Ten<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        tensor_format_with_formatter(self, "Ten", f)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::index::TensorIndex;
    use crate::prelude::*;
    use crate::{StableTorchResult, Tensor};

    #[test]
    fn test_flash_powder_debug_print() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([4,4])
        */

        let d = Tensor::from(&[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ])?;
        assert_eq!(d.sizes(), &[4, 4]); // #PYTHON list(d.shape)

        // 0d
        let sub_0d = d.i((0, 0))?;
        println!("sub_0d:\n{sub_0d:?}");
        let sub_1d = d.view(&[d.numel()])?;
        println!("sub_1d:\n{sub_1d:?}");

        println!("randn:\n{:?}", Tensor::randn(&[10], &Default::default())?);
        println!("randn:\n{:?}", Tensor::randn(&[5000], &Default::default())?);
        Ok(())
    }
}
