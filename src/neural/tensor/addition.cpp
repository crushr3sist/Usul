#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>

/// ADDITION
// tensor to tensor
Tensor Tensor::operator+(Tensor other) {

  vector<Tensor> children = {*this, other};

  Tensor result(this->data + other.data, children);

  auto *this_ptr = this;
  auto *other_ptr = &other;
  auto *result_ptr = &result;
  result.backwards = [this_ptr, other_ptr, result_ptr]() {

    this_ptr->gradients += result_ptr->gradients;
    other_ptr->gradients += result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr, other_ptr};
  Node n(OP::OP_ADD, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
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

  auto *this_ptr = this;
  auto *result_ptr = &result;

  result.backwards = [this_ptr, result_ptr]() {
    this_ptr->gradients += result_ptr->gradients;
  };

  // once everything is done, we need to make a node.
  vector<Tensor *> t = {this_ptr};
  Node n(OP::OP_ADD, false, result_ptr, t);
  // then we feed that node into our compute graph that stores it.
  State::ComputeGraph.publish(n);

  return result;
}

// 1.0 + tensor
Tensor operator+(double lhs, Tensor &rhs) {
  // get the result
  // we need to get rid of the children
  // i know that sounds like a wish but the new result shouldnt inherit a "temp" change to the compute graph.
  Tensor result(lhs + rhs.data, {});
  // the left handside is already a pointer
  // make a pointer to the right hand side

  Tensor *rhs_ptr = &rhs;
  // for the result, its just simple, we take the tensor and assin it the upstream gradient
  // but using accum.
  result.backwards = [rhs_ptr, &result]() {
    rhs_ptr->gradients += result.gradients;
  };
  // then we just create a pointer array of tensors

  vector<Tensor *> t = {rhs_ptr};
  // assign it to the nodes

  Node n(OP::OP_ADD, false, &result, t);

  // and publish it into the compute graph.

  State::ComputeGraph.publish(n);
  return result;
}
