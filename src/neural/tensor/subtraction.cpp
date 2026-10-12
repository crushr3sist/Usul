#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

Tensor Tensor::operator-(Tensor other) {
  Tensor result(this->data - other.data, children);

  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = result_ptr->gradients;
    other_ptr->gradients = result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_SUB, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}
Tensor Tensor::operator-(double other) {
  Tensor result(this->data - other.data, children);

  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = result_ptr->gradients;
    other_ptr->gradients = result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_SUB, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}
Tensor operator-(double lhs, Tensor &rhs) {
  Tensor result(lhs - rhs.data, {});

  auto *result_ptr = &result;
  auto *lhs_ptr = &lhs;

  result.backwards = [result_ptr]() {
    result->gradients = result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_SUB, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}