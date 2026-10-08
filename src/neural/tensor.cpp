#include "neural/tensor.hpp"
#include "neural/computegraph.hpp"
#include "neural/node.hpp"
#include "usul.hpp"
#include <memory>
#include <utility>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>

Tensor::Tensor(xt::xarray<double> data,
               vector<Tensor> children,
               xt::xarray<double> gradients) {
  this->data = data;
  this->children = std::move(children);
  this->gradients = gradients;
  this->backwards = []() {};
}

Tensor Tensor::operator+(Tensor other) {
  Tensor result(this->data + other.data, children);

  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = result_ptr->gradients;
    other_ptr->gradients = result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_ADD, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

Tensor Tensor::operator*(Tensor &other) {

  Tensor result(this->data * other.data, children);
  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = result_ptr->gradients * other_ptr->gradients;
    other_ptr->gradients = result_ptr->gradients * this_ptr->gradients;
  };
  return result;
}

Tensor Tensor::operator^(Tensor &other) {

  Tensor result(xt::linalg::dot(this->data, other.data), children);
  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = xt::linalg::dot(result_ptr->gradients, xt::transpose(other_ptr->gradients));
    other_ptr->gradients = xt::linalg::dot(xt::transpose(this_ptr->gradients), result_ptr->gradients);
  };
  return result;
}
