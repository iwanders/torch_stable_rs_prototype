use crate::core_methods::CoreMethods;
use crate::data::DataRef;
use crate::properties::TensorProperties;
use crate::tensor::{Ten, TenMut, Tensor, TensorAccess};

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

pub trait PrintRequirements: TensorAccess + TensorProperties + CoreMethods + DataRef {}
impl<T: TensorAccess + TensorProperties + CoreMethods + DataRef> PrintRequirements for T {}

// https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L130
//
// They do two passess, one to determine the width of the elements, then another to actually print.
fn tensor_format<T: PrintRequirements>(
    t: &T,
    indent: usize,
    options: &TensorPrintOptions,
    summarize: bool,
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

    let tensor_str = |indent: usize, o: &T, summarize: bool| -> String {
        let mut r = String::new();
        // Okay... the big one.
        let mut slices: Vec<String> = vec![];
        // https://github.com/pytorch/pytorch/blob/8f8409cae86d725a75e2ac54ce8f93def107ced7/torch/_tensor_str.py#L304
        if summarize && t.size(0) > 2 * options.print_options.edgeitems {
            slices.extend((0..options.print_options.edgeitems).map(|i| {
                tensor_format(
                    &crate::torch::select(t, 0, i).unwrap(),
                    indent,
                    options,
                    summarize,
                )
            }));
            slices.push("...".to_owned());
            slices.extend(
                (t.size(0) - (options.print_options.edgeitems)..t.size(0)).map(|i| {
                    tensor_format(
                        &crate::torch::select(t, 0, i).unwrap(),
                        indent,
                        options,
                        summarize,
                    )
                }),
            );
        } else {
            slices = (0..t.size(0))
                .map(|i| {
                    tensor_format(
                        &crate::torch::select(t, 0, i).unwrap(),
                        indent,
                        options,
                        summarize,
                    )
                })
                .collect();
        }
        r += "[";
        let joiner = format!(",{}{}", "\n".repeat(o.dim() - 1), " ".repeat(indent));
        r += &slices.join(&joiner);
        r += "]";
        r
    };

    if t.dim() == 0 {
        format!("{}", t.d_fmt(&[], &options.scalar_options).unwrap())
    } else if t.dim() == 1 {
        let element_width = determine_width(t);

        let mut r = String::new();
        r += &vector_str(indent + 1, t, element_width, summarize);
        r
    } else {
        let mut r = String::new();
        r += &tensor_str(indent + 1, t, summarize);
        r
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
    };

    let summarize = t.numel() > print_options.print_options.threshold;

    let mut r = format!("{}(", prefix);
    r += &tensor_format(t, r.len(), &print_options, summarize);
    r += ")";
    fmt.write_fmt(format_args!("{}", r))
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

        println!(
            "randn:\n{:?}",
            Tensor::randn(&[50, 50], &Default::default())?
        );
        println!(
            "randn:\n{:?}",
            Tensor::randn(&[50, 50, 10], &Default::default())?
        );

        Ok(())
    }
}
