#include "usul.hpp"
#include <xtensor/generators/xbuilder.hpp>

Tensor::Tensor(xt::xarray<double> data) {
  // generate the data struct and then install the data into it.
  this->heart = std::make_shared<Heart>();
  this->heart->data = std::move(data);
  this->heart->gradients = xt::zeros<double>(this->heart->data.shape());
  this->heart->backwards = []() {};
}

Tensor Tensor::operator-() {
  return Tensor(-this->heart->data);
}

Tensor Tensor::exponential() {
  return Tensor(xt::exp(this->heart->data));
}

Tensor Tensor::max(double value) {
  return Tensor(xt::maximum(this->heart->data, value));
}

Tensor Tensor::mean() {
  // we need the pointer for our funciotn

  auto this_ptr = this->heart;

  auto data = xt::mean(this->heart->data);
  Tensor result(data);

  result.heart->children = {this_ptr};
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {
      this_ptr->gradients += result_heart->gradients / static_cast<double>(this_ptr->data.size());
    }
  };

  Node n(OP::OP_MEAN, result, result.heart->children);
  State::ComputeGraph.publish(n);

  return result;
}

Tensor Tensor::sum() {
  return Tensor(xt::sum(this->heart->data));
}

Tensor Tensor::sum(std::vector<std::size_t> axes) {
  return Tensor(xt::sum(this->heart->data, axes, xt::keep_dims));
}

Tensor Tensor::transpose() {
  return Tensor(xt::transpose(this->heart->data));
}

// reshape CAN stay in-place if you want Y_true.reshape({4, 1}) to work as written:
Tensor Tensor::reshape(std::vector<int> shape) {
  this->heart->data.reshape(shape);
  return *this;
}

void Tensor::backwards() {

  // basically, this function should do 3 things
  // 1. it should seed the last entries gradients with the same shape as data but fill the gradients with 1
  this->heart->gradients = xt::ones<double>(this->heart->data.shape());

  // 2. iterate through the rest of the compute graph
  for (size_t i = State::UID; i > 0; i--) {
    auto heart = State::ComputeGraph.nodes.at(i).resultant_tensor.heart;

    // derive operation and create our gradients
    heart->backwards();
  }
  State::ComputeGraph.clear();
}

void Tensor::step(double LR) {

  // 3. modify the data by the gradients
  this->heart->data -= LR * heart->gradients;

  // 4. we then need to clear the gradients
  this->heart->gradients = xt::zeros<double>(heart->data.shape());
}

Tensor Tensor::grad() {
  return this->heart->gradients;
}
