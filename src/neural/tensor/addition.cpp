#include "usul.hpp"

// !NOTE - we're speeding up the functional definition of our arithmatics.
// and we're also figuring out the smart ptrs

/// ADDITION
// tensor to tensor

Tensor Tensor::operator+(const Tensor &other) const {
  // copy our this pointer
  auto this_ptr = this->heart;
  // copy our other pointer
  auto other_ptr = other.heart;
  // generating the tensor using minimal data
  Tensor result(this->heart->data + other.heart->data);

  // setting the children in
  result.heart->children = {this_ptr, other_ptr};
  // result can be chucked in because it is a conduit, it is a hollow utterance of the data that is hidden behind out heart struct shared_ptr

  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, other_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      auto this_local = result_heart->gradients;
      auto other_local = result_heart->gradients;

      if (other_ptr->data.dimension() > 0 && other_ptr->data.shape(0) == 1 && result_heart->data.shape(0) > 1) {

        other_local = xt::sum(result_heart->gradients, {0});
      }
      if (this_ptr->data.dimension() > 0 && this_ptr->data.shape(0) == 1 && result_heart->data.shape(0) > 1) {

        this_local = xt::sum(result_heart->gradients, {0});
      }

      this_ptr->gradients += this_local;
      other_ptr->gradients += other_local;
    }
  };

  Node n(OP::OP_ADD, result, result.heart->children);
  State::ComputeGraph.publish(n);

  return result;
}

// tensor + 1.0
/// correct.
Tensor Tensor::operator+(double other) const {
  auto this_ptr = this->heart;

  Tensor result(this->heart->data + other);

  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      this_ptr->gradients += result_heart->gradients;
    }
  };
  vector<shared_ptr<Heart>> t = {this_ptr};
  Node n(OP::OP_ADD, result, t);
  State::ComputeGraph.publish(n);
  return result;
}

// 1.0 + tensor
Tensor operator+(double lhs, const Tensor &rhs) {
  auto rhs_ptr = rhs.heart;
  Tensor result(lhs + rhs.heart->data);
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [rhs_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      rhs_ptr->gradients += result_heart->gradients;
  } };

  vector<shared_ptr<Heart>> t = {rhs_ptr};

  Node n(OP::OP_ADD, result, t);

  State::ComputeGraph.publish(n);
  return result;
}
