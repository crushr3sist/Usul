#include "usul.hpp"

/// ELEMENT-WISE MULTIPLICATION
// tensor to tensor
Tensor Tensor::operator*(const Tensor &other) const {
  auto this_ptr = this->heart;
  auto other_ptr = other.heart;

  Tensor result(this->heart->data * other.heart->data);
  result.heart->children = {this_ptr, other_ptr};
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, other_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      this_ptr->gradients += result_heart->gradients * other_ptr->data;
      other_ptr->gradients += result_heart->gradients * this_ptr->data;
    } };

  // NOTE - initing like this copies the tensor
  auto t = {this_ptr, other_ptr};
  Node n(OP::OP_HADAMARD, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

Tensor
Tensor::operator%(const Tensor &other) const {
  auto this_ptr = this->heart;
  auto other_ptr = other.heart;

  Tensor result(xt::linalg::dot(this->heart->data, other.heart->data));

  result.heart->children = {this_ptr, other_ptr};

  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, other_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      this_ptr->gradients += xt::linalg::dot(result_heart->gradients, xt::transpose(other_ptr->data));
      other_ptr->gradients += xt::linalg::dot(xt::transpose(this_ptr->data), result_heart->gradients);
  } };

  // once everything is done, we need to make a node.
  auto t = {this_ptr, other_ptr};
  Node n(OP::OP_DOT, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

// tensor * 1.0
Tensor Tensor::operator*(double other) const {
  auto this_ptr = this->heart;

  Tensor result(this->heart->data * other);
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, result_ptr, other]() {
    if (auto result_heart = result_ptr.lock()) {
      this_ptr->gradients += result_heart->gradients * other;
  } };
  // once everything is done, we need to make a node.
  auto t = {this_ptr};
  Node n(OP::OP_HADAMARD, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}

// 1.0 * tensor
Tensor operator*(double lhs, const Tensor &rhs) {
  auto rhs_ptr = rhs.heart;
  Tensor result(lhs * rhs.heart->data);
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [rhs_ptr, result_ptr, lhs]() {
    if (auto result_heart = result_ptr.lock()) {
      rhs_ptr->gradients += result_heart->gradients * lhs;
  } };
  // once everything is done, we need to make a node.
  auto t = {rhs_ptr};
  Node n(OP::OP_HADAMARD, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);
  return result;
}