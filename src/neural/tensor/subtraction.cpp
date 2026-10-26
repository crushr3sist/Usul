#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

// tensor + tensor
Tensor Tensor::operator-(Tensor &other) {
  auto this_ptr = this->heart;
  auto other_ptr = other.heart;

  Tensor result(this->heart->data - other.heart->data);

  result.heart->backwards = [this_ptr, other_ptr, result]() {
    this_ptr->gradients += result.heart->gradients;
    other_ptr->gradients -= result.heart->gradients;
  };

  // NOTE - initing like this copies the tensor
  auto t = {this_ptr, other_ptr};
  Node n(OP::OP_SUB, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// tensor + 1.0
Tensor Tensor::operator-(double other) {
  auto this_ptr = this->heart;

  Tensor result(this->heart->data - other);

  result.heart->backwards = [this_ptr, result]() {
    this_ptr->gradients += result.heart->gradients;
  };
  // once everything is done, we need to make a node.
  auto t = {this_ptr};

  Node n(OP::OP_SUB, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// 1.0 + tensor
Tensor operator-(double lhs, Tensor &rhs) {
  auto rhs_ptr = rhs.heart;

  Tensor result(lhs - rhs.heart->data);

  result.heart->backwards = [rhs_ptr, result]() {
    rhs_ptr->gradients += -result.heart->data;
  };

  // once everything is done, we need to make a node.
  auto t = {rhs_ptr};
  Node n(OP::OP_SUB, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}