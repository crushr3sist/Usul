#include "neural/tensor.hpp"
#include <utility>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <memory>
#include <xtensor/misc/xmanipulation.hpp>

Tensor::Tensor(xt::xarray<double> data) {
  // generate the data struct and then install the data into it.
  this->heart = std::make_shared<Heart>();
  this->heart->data = std::move(data);
  this->heart->backwards = []() {};
}

Tensor Tensor::operator-() {
  return Tensor(-this->heart->data);
}

Tensor Tensor::exponential() {
  return Tensor(xt::exp(this->heart->data));
}

Tensor Tensor::max(double value) {
  return Tensor(xt::maximum(this->heart->data, value));
}

Tensor Tensor::mean() {
  return Tensor(xt::mean(this->heart->data));
}

Tensor Tensor::sum() {
  return Tensor(xt::sum(this->heart->data));
}

Tensor Tensor::sum(std::vector<std::size_t> axes) {
  return Tensor(xt::sum(this->heart->data, axes, xt::keep_dims));
}

Tensor Tensor::transpose() {
  return Tensor(xt::transpose(this->heart->data));
}

// reshape CAN stay in-place if you want Y_true.reshape({4, 1}) to work as written:
Tensor Tensor::reshape(std::vector<int> shape) {
  this->heart->data.reshape(shape);
  return *this;
}