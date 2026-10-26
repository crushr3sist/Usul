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
  // // then we feed that node into our compute graph that stores it.
  // State::ComputeGraph.publish(n);

  // return result;
}

// tensor + 1.0
/// correct.
Tensor Tensor::operator+(double other) {

  // scalar to matrix addition
  // xtensor does alot of the heavy lifting for us
  // we can just do this->data + other due to xtensor supporting broadcasting
  Tensor result(this->data + other, {});
  // then we make a pointer to our tensor
  // and then a pointer to the resultant tensor

  shared_ptr<Tensor> this_safe = make_shared<Tensor>(*this);
  shared_ptr<Tensor> result_ptr = make_shared<Tensor>(result);

  result.backwards = [this_safe, result_ptr]() {
    this_safe->gradients += result_ptr->gradients;
  };

  // NOTE - initing like this copies the tensor
  vector<shared_ptr<Tensor>> t = {this_safe};
  Node n(OP::OP_ADD, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// 1.0 + tensor
Tensor operator+(double lhs, Tensor &rhs) {

  Tensor result(lhs + rhs.data, {});

  shared_ptr<Tensor> rhs_ptr = make_shared<Tensor>(rhs);
  shared_ptr<Tensor> result_ptr = make_shared<Tensor>(result);

  result.backwards = [rhs_ptr, result_ptr]() {
    rhs_ptr->gradients += result_ptr->gradients;
  };

  vector<shared_ptr<Tensor>> t = {rhs_ptr};

  Node n(OP::OP_ADD, false, result_ptr, t);

  State::ComputeGraph.publish(n);
  return result;
}
