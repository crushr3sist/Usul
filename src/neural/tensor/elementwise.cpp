#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

/// ELEMENT-WISE MULTIPLICATION
// tensor to tensor
Tensor Tensor::operator*(Tensor other) {
  vector<Tensor> children = {*this, other};

  Tensor result(this->data * other.data, children);

  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients += result_ptr->gradients * other_ptr->data;
    other_ptr->gradients += result_ptr->gradients * this_ptr->data;
  };
  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_HADAMARD, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

// tensor * 1.0
Tensor Tensor::operator*(double other) {

  Tensor result(this->data * other, {});
  auto *this_ptr = this;
  auto *result_ptr = &result;
  auto *other_ptr = &other;

  result.backwards = [this_ptr, result_ptr, other_ptr]() {
    this_ptr->gradients += result_ptr->gradients * *other_ptr;
  };
  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr};
  Node n(OP::OP_HADAMARD, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

// 1.0 * tensor
Tensor operator*(double lhs, Tensor &rhs) {

  Tensor result(lhs * rhs.data, {});
  auto *rhs_ptr = &rhs;
  auto *res_ptr = &result;
  auto *lhs_ptr = &lhs;
  result.backwards = [rhs_ptr, res_ptr, lhs_ptr]() {
    rhs_ptr->gradients += res_ptr->gradients * *lhs_ptr;
  };
  // once everything is done, we need to make a node.
  vector<Tensor *> t = {rhs_ptr};
  Node n(OP::OP_HADAMARD, false, res_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}