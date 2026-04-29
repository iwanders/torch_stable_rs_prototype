

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>

#include <optional>
#include <iostream>

using torch::stable::Tensor;


void run_torch_stable_empty() {
  std::cout << "suppression@torch::stable::empty" << std::endl;
  torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{1,1,4,4}, {}, {}, {}, {}, {});
}
void run_torch_stable_to() {
  std::cout << "suppression@torch::stable::to" << std::endl;
  const auto t = torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{1,1,4,4}, {}, {}, {}, {}, {});
  // https://github.com/pytorch/pytorch/blob/2deace1f225e3ff25d776971def33ca59209c60a/torch/csrc/stable/ops.h#L824
  torch::stable::to(t, {torch::headeronly::ScalarType::Float} );
}


int main(int argc, char* argv[]){
  run_torch_stable_empty();
  run_torch_stable_to();
  return 0;
}
