#include "neural/tensor.hpp"
#include <utility>
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

// Tensor Tensor::operator^(Tensor &other) {

//   Tensor result(xt::linalg::dot(this->data, other.data), children);
//   auto *this_ptr = this;
//   auto *other_ptr = &other;
//   auto *result_ptr = &result;

//   result.backwards = [this_ptr, other_ptr, result_ptr]() {
//     this_ptr->gradients = xt::linalg::dot(result_ptr->gradients, xt::transpose(other_ptr->gradients));
//     other_ptr->gradients = xt::linalg::dot(xt::transpose(this_ptr->gradients), result_ptr->gradients);
//   };

//   // once everything is done, we need to make a node.
//   vector<Tensor *> t = {this_ptr, other_ptr};
//   Node n(OP::OP_ADD, false, result_ptr, t);
//   // then we feed that node into our compute graph that stores it.
//   State::ComputeGraph.publish(n);

//   return result;
// }

// Tensor Tensor::operator-() {
//   this->data = -this->data;
//   return *this;
// };

// Tensor Tensor::exponential() {
//   this->data = xt::exp(this->data);
//   return *this;
// }

// Tensor Tensor::max(double value) {}
// Tensor Tensor::mean() {}
// Tensor Tensor::reshape(vector<int> &shape) {}
