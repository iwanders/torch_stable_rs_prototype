

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>

#include <optional>

using torch::stable::Tensor;


void run_use_empty() {
  torch::stable::empty(torch::headeronly::HeaderOnlyArrayRef<int64_t>{1,1,4,4}, {}, {}, {}, {}, {});
}

int main(int argc, char* argv[]){
  run_use_empty();
  return 0;
}
