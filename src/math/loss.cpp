
#include "neural/tensor.hpp"

Tensor Tensor::MSE(const Tensor &Y_true) {
  auto distance = *this - Y_true;
  auto loss = (distance * distance).mean();
  return loss;
}

// Tensor Tensor::MSE_Prime() {}

// Tensor Tensor::BGD() {
// }
// Tensor Tensor::BGD_prime() {}

// Tensor Tensor::SGD() {
// }
// Tensor Tensor::SGD_prime() {}