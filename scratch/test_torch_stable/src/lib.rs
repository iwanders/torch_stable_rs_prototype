// https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h#L21

unsafe extern "C" {
    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L56
    fn aoti_torch_device_type_cuda() -> i32;
    fn aoti_torch_device_type_cpu() -> i32;
}

pub fn main() {
    unsafe {
        println!("Hello, world!");
        println!(
            "aoti_torch_device_type_cuda: {}",
            aoti_torch_device_type_cuda()
        );
        println!(
            "aoti_torch_device_type_cpu: {}",
            aoti_torch_device_type_cpu()
        );
    }
}
