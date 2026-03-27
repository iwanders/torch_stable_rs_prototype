use anyhow::anyhow;
// https://github.com/pytorch/pytorch/tree/fbdef9635b009f670321b1263bec7b48e2d7379f/torch/headeronly/core

// https://github.com/pytorch/pytorch/blob/fbdef9635b009f670321b1263bec7b48e2d7379f/torch/headeronly/core/DeviceType.h#L35
// Should this be in headeronly/core/DeviceType?
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum DeviceType {
    CPU = 0,
    CUDA = 1,         // CUDA.
    MKLDNN = 2,       // Reserved for explicit MKLDNN
    OPENGL = 3,       // OpenGL
    OPENCL = 4,       // OpenCL
    IDEEP = 5,        // IDEEP.
    HIP = 6,          // AMD HIP
    FPGA = 7,         // FPGA
    MAIA = 8,         // ONNX Runtime / Microsoft
    XLA = 9,          // XLA / TPU
    Vulkan = 10,      // Vulkan
    Metal = 11,       // Metal
    XPU = 12,         // XPU
    MPS = 13,         // MPS
    Meta = 14,        // Meta (tensors with no data)
    HPU = 15,         // HPU / HABANA
    VE = 16,          // SX-Aurora / NEC
    Lazy = 17,        // Lazy Tensors
    IPU = 18,         // Graphcore IPU
    MTIA = 19,        // Meta training and inference devices
    PrivateUse1 = 20, // PrivateUse1 device
}

impl TryFrom<u32> for DeviceType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            v if v == (DeviceType::CPU as u32) => Ok(DeviceType::CPU),
            v if v == (DeviceType::CUDA as u32) => Ok(DeviceType::CUDA),
            v if v == (DeviceType::MKLDNN as u32) => Ok(DeviceType::MKLDNN),
            v if v == (DeviceType::OPENGL as u32) => Ok(DeviceType::OPENGL),
            v if v == (DeviceType::OPENCL as u32) => Ok(DeviceType::OPENCL),
            v if v == (DeviceType::IDEEP as u32) => Ok(DeviceType::IDEEP),
            v if v == (DeviceType::HIP as u32) => Ok(DeviceType::HIP),
            v if v == (DeviceType::FPGA as u32) => Ok(DeviceType::FPGA),
            v if v == (DeviceType::MAIA as u32) => Ok(DeviceType::MAIA),
            v if v == (DeviceType::XLA as u32) => Ok(DeviceType::XLA),
            v if v == (DeviceType::Vulkan as u32) => Ok(DeviceType::Vulkan),
            v if v == (DeviceType::Metal as u32) => Ok(DeviceType::Metal),
            v if v == (DeviceType::XPU as u32) => Ok(DeviceType::XPU),
            v if v == (DeviceType::MPS as u32) => Ok(DeviceType::MPS),
            v if v == (DeviceType::Meta as u32) => Ok(DeviceType::Meta),
            v if v == (DeviceType::HPU as u32) => Ok(DeviceType::HPU),
            v if v == (DeviceType::VE as u32) => Ok(DeviceType::VE),
            v if v == (DeviceType::Lazy as u32) => Ok(DeviceType::Lazy),
            v if v == (DeviceType::IPU as u32) => Ok(DeviceType::IPU),
            v if v == (DeviceType::MTIA as u32) => Ok(DeviceType::MTIA),
            v if v == (DeviceType::PrivateUse1 as u32) => Ok(DeviceType::PrivateUse1),
            _ => Err(anyhow!("could not convert {} into DeviceType", value)),
        }
    }
}
