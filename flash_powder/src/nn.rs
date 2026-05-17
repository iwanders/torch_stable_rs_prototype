//! Torch's nn module.
//!
//! <https://docs.pytorch.org/docs/2.12/nn.html>
//!
//! This is not exposed at all through the stable API, so this is a pure rust implementation.

use torch_stable::StableTorchResult;

use crate::{Ten, Tensor, core_methods::CoreMethods};

#[derive(Debug, Clone)]
pub enum Data {
    /// Tensors that record gradients, typically weights.
    Parameter(Tensor),
    /// Tensors that do not record gradients, updated during forward step.
    Buffer(Tensor),
    // State;  The non-tensor state :/
    // maybe a Box<dyn Any> ?
}

#[derive(Debug, Clone, Default)]
pub struct StateDict {
    map: std::collections::HashMap<String, Data>,
}
impl StateDict {
    pub fn add_parameter(&mut self, name: &str, value: Tensor) -> StableTorchResult<()> {
        self.add_data(name, Data::Parameter(value))
    }
    pub fn add_buffer(&mut self, name: &str, value: Tensor) -> StableTorchResult<()> {
        self.add_data(name, Data::Buffer(value))
    }
    pub fn add_data(&mut self, name: &str, value: Data) -> StableTorchResult<()> {
        self.map
            .insert(name.to_owned(), value)
            .ok_or(anyhow::anyhow!("entry with name {name} already existed"))
            .map(|_| ())
    }
    pub fn add_state_dict(&mut self, name: &str, mut value: StateDict) -> StableTorchResult<()> {
        // Munge paths and concat.
        for (k, v) in value.map.drain() {
            let new_name = format!("{name}.{k}");
            self.add_data(&new_name, v)?;
        }
        Ok(())
    }
    pub fn as_map(&self) -> &std::collections::HashMap<String, Data> {
        &self.map
    }
    pub fn as_map_mut(&mut self) -> &mut std::collections::HashMap<String, Data> {
        &mut self.map
    }
}

pub trait StateDictAdaptor {
    fn tensor(&self, name: &str) -> Option<Tensor>;
}
impl StateDictAdaptor for StateDict {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        if let Some(record) = self.map.get(name) {
            match record {
                Data::Parameter(tensor) => Some(tensor.clone()),
                Data::Buffer(tensor) => Some(tensor.clone()),
            }
        } else {
            None
        }
    }
}

/// Base trait for all neural network modules.
///
/// - Pytorch Docs: <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html>
/// - C++ docs: <https://docs.pytorch.org/cppdocs/api/nn/index.html#module-base-class>
/// - Python class code: <https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/module.py#L407>
/// - C++ class code: <https://github.com/pytorch/pytorch/blob/v2.12.0/torch/csrc/api/include/torch/nn/module.h#L63>
///
/// Three kinds of persistent data (from c++ docs):
/// - Parameters; Tensors that record gradient, typically weights, like `weight` of linear.
/// - Buffers; Tensor that do not record gradients, typically updated during forward step; `mean`, `variance` of BatchNorm.
/// - Additionally state, not necessarily tensors, required for implementation or configuration of a Module.
pub trait Module: std::fmt::Debug + dyn_clone::DynClone {
    fn forward(&self, v: &Ten<'_>) -> Result<Tensor, anyhow::Error>;
    // These look relevant;
    // register_buffer
    // register_parameter
    // add_module / register_module
    // get_submodule
    // set_submodule
    // get_parameter
    // get_buffer
    // get_extra_state
    // set_extra_state
    // apply
    // to
    // __call__
    // __setstate__
    // __getstate__
    // state_dict
    // load_state_dict
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        Ok(Default::default())
    }
    fn load_state_dict(&mut self, dict: &dyn StateDictAdaptor) -> StableTorchResult<()> {
        let _ = dict;
        Ok(())
    }
}

dyn_clone::clone_trait_object!(Module);

/// Sequential module
///
/// - pytorch equivalent; <https://docs.pytorch.org/docs/2.12/generated/torch.nn.Sequential.html>
#[derive(Debug, Clone)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}
impl Module for Sequential {
    fn forward(&self, v: &Ten<'_>) -> Result<Tensor, anyhow::Error> {
        if self.modules.is_empty() {
            return Ok(v.to_owned()?);
        }
        let mut intermediate = self.modules.first().unwrap().forward(v)?;
        for remaining_layers in self.modules.iter().skip(1) {
            intermediate = remaining_layers.forward(&intermediate.ten()?)?;
        }
        Ok(intermediate)
    }
    fn state_dict(&self) -> StableTorchResult<StateDict> {
        let mut m: StateDict = Default::default();
        for (i, v) in self.modules.iter().enumerate() {
            let name = format!("{i}");
            m.add_state_dict(&name, v.state_dict()?)?
        }
        Ok(m)
    }
}
