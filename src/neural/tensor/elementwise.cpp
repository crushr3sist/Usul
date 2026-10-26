#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

/// ELEMENT-WISE MULTIPLICATION
// tensor to tensor
Tensor Tensor::operator*(Tensor &other) {
  auto this_ptr = this->heart;
  auto other_ptr = other.heart;

  Tensor result(this->heart->data * other.heart->data);
  result.heart->children = {this_ptr, other_ptr};

  result.heart->backwards = [this_ptr, other_ptr, result]() {
    this_ptr->gradients += result.heart->gradients * other_ptr->data;
    other_ptr->gradients += result.heart->gradients * this_ptr->data;
  };

  // NOTE - initing like this copies the tensor
  auto t = {this_ptr, other_ptr};
  Node n(OP::OP_HADAMARD, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

// tensor * 1.0
Tensor Tensor::operator*(double other) {
  auto this_ptr = this->heart;

  Tensor result(this->heart->data * other);

  result.heart->backwards = [this_ptr, result, &other]() {
    this_ptr->gradients += result.heart->gradients * other;
  };
  // once everything is done, we need to make a node.
  auto t = {this_ptr};
  Node n(OP::OP_HADAMARD, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

// 1.0 * tensor
Tensor operator*(double lhs, Tensor &rhs) {
  auto rhs_ptr = rhs.heart;
  Tensor result(lhs * rhs.heart->data);

  result.heart->backwards = [rhs_ptr, result, &lhs]() {
    rhs_ptr->gradients += result.heart->gradients * lhs;
  };
  // once everything is done, we need to make a node.
  auto t = {rhs_ptr};
  Node n(OP::OP_HADAMARD, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}