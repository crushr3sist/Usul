#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

// tensor + tensor
Tensor Tensor::operator-(const Tensor &other) const {
  auto this_ptr = this->heart;
  auto other_ptr = other.heart;

  Tensor result(this->heart->data - other.heart->data);
  result.heart->children = {this_ptr, other_ptr};

  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, other_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      // here we need to check if either tensor was broadcasted

      auto this_local = result_heart->gradients;
      auto other_local = result_heart->gradients;

      if (other_ptr->data.dimension() > 0 && other_ptr->data.shape(0) == 1 && result_heart->data.shape(0) > 1) {

        other_local = xt::sum(result_heart->gradients, {0});
      }
      if (this_ptr->data.dimension() > 0 && this_ptr->data.shape(0) == 1 && result_heart->data.shape(0) > 1) {

        this_local = xt::sum(result_heart->gradients, {0});
      }

      this_ptr->gradients += this_local;
      other_ptr->gradients -= other_local;
    }
  };

  // NOTE - initing like this copies the tensor
  auto t = {this_ptr, other_ptr};
  Node n(OP::OP_SUB, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// tensor + 1.0
Tensor Tensor::operator-(double other) const {
  auto this_ptr = this->heart;

  Tensor result(this->heart->data - other);
  result.heart->children = {this_ptr};
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      this_ptr->gradients += result_heart->gradients;
  } };
  // once everything is done, we need to make a node.
  auto t = {this_ptr};

  Node n(OP::OP_SUB, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// 1.0 + tensor
Tensor operator-(double lhs, const Tensor &rhs) {
  auto rhs_ptr = rhs.heart;

  Tensor result(lhs - rhs.heart->data);
  result.heart->children = {rhs_ptr};
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [rhs_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      rhs_ptr->gradients += -result_heart->gradients;
    } };

  // once everything is done, we need to make a node.
  auto t = {rhs_ptr};
  Node n(OP::OP_SUB, false, result, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}