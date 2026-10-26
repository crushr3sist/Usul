#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

// tensor + tensor
Tensor Tensor::operator-(Tensor &other) {

  shared_ptr<Tensor> this_safe = make_shared<Tensor>(*this);
  shared_ptr<Tensor> other_safe = make_shared<Tensor>(other);
  vector<shared_ptr<Tensor>> children;
  children.emplace_back(this_safe);
  children.emplace_back(other_safe);

  Tensor result(this->data - other.data, children);

  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients += result_ptr->gradients;
    other_ptr->gradients -= result_ptr->gradients;
  };

  // NOTE - initing like this copies the tensor
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_SUB, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// tensor + 1.0
Tensor Tensor::operator-(double other) {
  Tensor result(this->data - other, {});

  auto *this_ptr = this;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, result_ptr]() {
    this_ptr->gradients += result_ptr->gradients;
  };
  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr};

  Node n(OP::OP_SUB, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// 1.0 + tensor
Tensor operator-(double lhs, Tensor &rhs) {
  Tensor result(lhs - rhs.data, {});

  auto *result_ptr = &result;
  auto *rhs_ptr = &rhs;

  result.backwards = [rhs_ptr, result_ptr]() {
    rhs_ptr->gradients += -result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {rhs_ptr};
  Node n(OP::OP_SUB, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}