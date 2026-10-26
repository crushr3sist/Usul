#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

// !NOTE - we're speeding up the functional definition of our arithmatics.
// and we're also figuring out the smart ptrs

/// ADDITION
// tensor to tensor
Tensor Tensor::operator+(Tensor &other) {
  // copy our this pointer
  auto this_ptr = this->heart;
  // copy our other pointer
  auto other_ptr = other.heart;
  // generating the tensor using minimal data
  Tensor result(this->heart->data + other.heart->data);

  // setting the children in
  result.heart->children = {this_ptr, other_ptr};
  // result can be chucked in because it is a conduit, it is a hollow utterance of the data that is hidden behind out heart struct shared_ptr

  result.heart->backwards = [this_ptr, other_ptr, result]() {
    this_ptr->gradients += result.heart->gradients;
    other_ptr->gradients += result.heart->gradients;
  };

  Node n(OP::OP_ADD, false, result, result.heart->children);
  State::ComputeGraph.publish(n);

  return result;
}

// tensor + 1.0
/// correct.
Tensor Tensor::operator+(double other) {
  auto this_ptr = this->heart;

  Tensor result(this->heart->data + other);

  result.heart->backwards = [this_ptr, result]() {
    this_ptr->gradients += result.heart->gradients;
  };
  vector<shared_ptr<Heart>> t = {this_ptr};
  Node n(OP::OP_ADD, false, result, t);
  State::ComputeGraph.publish(n);
  return result;
}

// 1.0 + tensor
Tensor operator+(double lhs, Tensor &rhs) {
  auto rhs_ptr = rhs.heart;
  Tensor result(lhs + rhs.heart->data);

  result.heart->backwards = [rhs_ptr, result]() {
    rhs_ptr->gradients += result.heart->gradients;
  };

  vector<shared_ptr<Heart>> t = {rhs_ptr};

  Node n(OP::OP_ADD, false, result, t);

  State::ComputeGraph.publish(n);
  return result;
}
