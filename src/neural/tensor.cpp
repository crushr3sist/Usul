#include "neural/tensor.hpp"
#include <memory>
#include <utility>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>

Tensor::Tensor(xt::xarray<double> data, vector<shared_ptr<Tensor>> children, xt::xarray<double> gradients) {
  this->data = data;
  this->children = std::move(children);
  this->gradients = gradients;
  this->backwards = []() {};
}

shared_ptr<Tensor> Tensor::create(xt::xarray<double> data, vector<shared_ptr<Tensor>> children, xt::xarray<double> gradients) {
  // we're moving the operation and children, as we do not need thier copies, they can belong to this node's instance
  shared_ptr<Tensor> new_node = shared_ptr<Tensor>(new Tensor(std::move(data), std::move(children), std::move(gradients)));
  return new_node;
}

shared_ptr<Tensor> Tensor::operator+(Tensor &other) {
  // NOTE - we need to know that, when doing addition, shapes need to be the exact same shape
  // we need to add a dimenstional checker and corrector.
  // if either matrix is uneven, we need to make sure we broadcast them
  // we also need to make a flag saying that this shape was broadcasted
  bool broadcasted = false;
  vector<shared_ptr<Tensor>> children;
  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);
  // we need to check to see if the matrix dimensions line up.

  if (this_ptr->data.shape() == other_ptr->data.shape()) {
    // we know both of the axis match up, its all good to go.
    println("same shape");
  } else {
    println("not the same shape");
    broadcasted = true;
  }
  shared_ptr<Tensor> result = Tensor::create(this->data + other.data, children);

  auto *result_ptr = result.get();

  result->backwards = [this_ptr, other_ptr, result_ptr, broadcasted]() {
    if (broadcasted) {
      // we know if we broadcasted, we can just check during back propagation
      // we check to see. if either of our axis match
      // if two match, then we can check to see which one is broadcasted.

      // enforce the addition rules
      // rule 1, check if both the left-most axis match
      // and if the either right most element is a 1

      if (this_ptr->data.shape()[0] == 1 || this_ptr->data.shape()[1] == 1) {
        // we know this_ptr has a 1 in its dimensions
        if (this_ptr->data.shape()[0] == 1) {
          // dimension 0 is a 1
        } else {
          // dimension 1 is a 1
        }
      }
      if (other_ptr->data.shape()[0] == 1 || other_ptr->data.shape()[1] == 1) {
        // we know other_ptr has a 1 in its dimensions
        if (other_ptr->data.shape()[0] == 1) {
          // dimension 0 is a 1
        }
        if (other_ptr->data.shape()[1] == 1) {
          // dimension 1 is a 1
        }
      }

    } else {
      this_ptr->gradients = result_ptr->gradients;
      other_ptr->gradients = result_ptr->gradients;
    }
  };
  return result;
}

shared_ptr<Tensor> Tensor::operator*(Tensor &other) {
  vector<shared_ptr<Tensor>> children;
  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  shared_ptr<Tensor> result = Tensor::create(this->data * other.data, children);

  auto *result_ptr = result.get();

  result->backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = result_ptr->gradients * other_ptr->gradients;
    other_ptr->gradients = result_ptr->gradients * this_ptr->gradients;
  };
  return result;
}
shared_ptr<Tensor> Tensor::operator^(Tensor &other) {
  vector<shared_ptr<Tensor>> children;
  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  shared_ptr<Tensor> result = Tensor::create(xt::linalg::dot(this->data, other.data), children);

  auto *result_ptr = result.get();

  result->backwards = [this_ptr, other_ptr, result_ptr]() {
    this_ptr->gradients = xt::linalg::dot(result_ptr->gradients, xt::transpose(other_ptr->gradients));
    other_ptr->gradients = xt::linalg::dot(xt::transpose(this_ptr->gradients), result_ptr->gradients);
  };
  return result;
}
