#include "neural/tensor.hpp"
#include <utility>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <memory>

Tensor::Tensor(xt::xarray<double> data) {
  // generate the data struct and then install the data into it.
  this->heart = std::make_shared<Heart>();
  this->heart->data = std::move(data);
  this->heart->backwards = []() {};
}

Tensor Tensor::operator-() {
  this->heart->data = -this->heart->data;
  return *this;
}

Tensor Tensor::exponential() {
  this->heart->data = xt::exp(this->heart->data);
  return *this;
}

Tensor Tensor::max(double value) {
  this->heart->data = xt::maximum(this->heart->data, value);
  return *this;
}

Tensor Tensor::mean() {
  this->heart->data = xt::mean(this->heart->data);
  return *this;
}

Tensor Tensor::reshape(std::vector<int> shape) {
  this->heart->data.reshape(shape);
  return *this;
}