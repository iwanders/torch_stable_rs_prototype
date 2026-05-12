// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py

// cfgs: dict[str, list[Union[str, int]]] = {
//     "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
//     "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
//     "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
//     "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
// }

// Smallest is vgg11;
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L306-L329
// which is of the 'A' category.
//    weights = VGG11_Weights.verify(weights)
// return _vgg("A", False, weights, progress, **kwargs)
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L98
// VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
// batch norm = False
// Layers per 'A'.

use anyhow::bail;
use safetensors::SafeTensors;
use std::{fs::File, path::Path};

use flash_powder as fp;
use flash_powder::functional;
use flash_powder::prelude::*;

trait ForwardLayer {
    fn forward(&self, tensor: &Tensor) -> Result<Tensor, anyhow::Error>;
}
type ForwardFun = dyn Fn(&Tensor) -> Result<Tensor, anyhow::Error>;
struct LambdaForward {
    f: Box<ForwardFun>,
}
impl LambdaForward {
    fn new<F>(f: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + 'static,
    {
        Self { f: Box::new(f) }
    }
}
impl ForwardLayer for LambdaForward {
    fn forward(&self, tensor: &Tensor) -> Result<Tensor, anyhow::Error> {
        (self.f)(tensor)
    }
}

// Gross, but it's uniformly integers now.
const CFG_A: &[u32] = &[
    64, 'M' as u32, 128, 'M' as u32, 256, 256, 'M' as u32, 512, 512, 'M' as u32, 512, 512,
    'M' as u32,
];

struct VGG {
    layers: Vec<Box<dyn ForwardLayer>>,
    classifier: Vec<Box<dyn ForwardLayer>>,
}
impl VGG {
    pub fn new(cfg: &[u32], tensors: &SafeTensors) -> Result<Self, anyhow::Error> {
        const BATCH_NORM: bool = false;
        let mut sorted = tensors.names().clone();
        sorted.sort();
        for name in sorted {
            println!("Tensor key: {}", name);

            // 3. (Optional) Get tensor info/data
            if let Ok(view) = tensors.tensor(name) {
                println!("  Shape: {:?}", view.shape());
            }
        }

        // Okay... so we iterate over cfg... build the layers and then append them to ourselves? How hard can it be.
        //
        let mut layers: Vec<Box<dyn ForwardLayer>> = vec![];
        let in_channels = 3;

        // First, the group called 'features'.
        let mut feature_counter = 0;
        for v in cfg.iter() {
            let layer_name = format!("features.{feature_counter}");
            if *v == 'M' as u32 {
                // should be maxpool2d.
                layers.push(Box::new(LambdaForward::new(|t: &Tensor| Ok(t.clone()))));
                feature_counter += 1;
            } else {
                let conv2d_options = functional::Conv2dOptions {
                    stride: (1, 1),
                    padding: (1, 1),
                    ..Default::default()
                };
                //
                let weights = { safetensor_to_tensor(tensors, &format!("{layer_name}.weight"))? };
                let bias = {
                    if let Ok(v) = safetensor_to_tensor(tensors, &format!("{layer_name}.bias")) {
                        Some(v)
                    } else {
                        None
                    }
                };
                layers.push(Box::new(LambdaForward::new(move |t: &Tensor| {
                    let bias_ref = bias.as_ref();
                    let conv_res = functional::conv2d(t, &weights, bias_ref, &conv2d_options)?;
                    let res = functional::relu(&conv_res)?;

                    Ok(res)
                })));
                // + 1 for conv2d, +1 for relu.
                feature_counter += 2;
            }
        }

        // and then the group called classifier
        let mut classifier: Vec<Box<dyn ForwardLayer>> = vec![];
        let mut classifier_counter = 0;
        let layer_name = format!("classifier.{classifier_counter}");

        Ok(VGG { layers, classifier })
    }
}

fn create_linear(
    tensors: &SafeTensors,
    name: &str,
) -> Result<Box<dyn ForwardLayer>, anyhow::Error> {
    todo!();
}

fn safetensor_dtype_to_scalar_type(v: safetensors::Dtype) -> fp::ScalarType {
    match v {
        safetensors::Dtype::F32 => fp::ScalarType::Float,
        safetensors::Dtype::F64 => fp::ScalarType::Double,
        _ => todo!("todo handle {v:?}"),
    }
}

fn safetensor_to_tensor(tensors: &SafeTensors, name: &str) -> Result<Tensor, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        // Create a tensor of the correct shape and type
        let mut v = Tensor::zeros(
            tensor_view.shape(),
            &flash_powder::factory::TensorOptions {
                dtype: Some(safetensor_dtype_to_scalar_type(tensor_view.dtype())),
                ..Default::default()
            },
        )?;

        // Copy the bytes.
        v.u8s_mut()?.copy_from_slice(tensor_view.data());
        Ok(v)
    } else {
        bail!("could not find safetensor {name}")
    }
}

fn dump_safensors(filepath: &Path) -> Result<(), anyhow::Error> {
    let data = std::fs::read(filepath).expect("Unable to read file");

    // 3. Parse the safetensors header
    let tensors = SafeTensors::deserialize(&data)?;

    // 2. Iterate over the keys (names)
    for name in tensors.names() {
        println!("Tensor key: {}", name);

        // 3. (Optional) Get tensor info/data
        if let Ok(view) = tensors.tensor(name) {
            println!("  Shape: {:?}", view.shape());
        }
    }
    /*
    // 4. Retrieve a specific tensor by name
    let tensor_view = tensors.tensor("weight_name")?;

    println!("Shape: {:?}", tensor_view.shape());
    println!("Dtype: {:?}", tensor_view.dtype());
    // Access raw data as &[u8]
    let data = tensor_view.data();*/

    Ok(())
}

pub fn main() -> Result<(), anyhow::Error> {
    use std::path::PathBuf;
    let weights = PathBuf::from("data/vgg11-8a719046.safetensors");
    if !weights.is_file() {
        eprintln!(
            "Run this binary from the 'example_vgg' directory, it looks for  \
            {}, if that doesn't exist:\n Download it from https://download.pytorch.org/models/vgg11-8a719046.pth,\
            convert it to safetensors with ./convert_pth.py",
            weights.display()
        );
        //
        bail!("missing necessary file, bailing out")
    }

    // dump_safensors(&weights)?;

    let data = std::fs::read(weights).expect("Unable to read file");

    // 3. Parse the safetensors header
    let tensors = SafeTensors::deserialize(&data)?;

    let vgg = VGG::new(&CFG_A, &tensors)?;

    Ok(())
}
