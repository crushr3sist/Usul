#include "usul.hpp"

Tensor generate_random_matrix(const std::vector<int> &shape, double low, double high) {
  auto random_expression = xt::random::rand<double>(shape, low, high);
  Tensor _(random_expression);
  return _;
}

Tensor Tensor::ReLU() {
  // we are setting the current state of our node
  // to relu mode
  // once we've relu'd the current node, we're then going to return a new tensor
  // we should return a new tensor.
  auto this_ptr = this->heart;

  auto data = xt::maximum(0.0, this->heart->data);
  Tensor result(data);
  // we still need to add it to the compute graph
  result.heart->children = {this_ptr};

  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {

      // so the derivative is i think,
      // PL/P_result (*) R;
      // we've got the relu prime
      auto relu_prime = xt::cast<double>(result_heart->data > 0.0);

      // now we need the upstream gradients.
      auto upstream_grad = result_heart->gradients;
      this_ptr->gradients += (upstream_grad * relu_prime);
    } };
  Node n(OP::OP_RELU, result, result.heart->children);

  State::ComputeGraph.publish(n);

  return result;
}

Tensor Tensor::ReLU_prime() {
  auto this_ptr = this->heart;

  auto data = xt::cast<double>(this->heart->data > 0.0);
  Tensor result(data);
  result.heart->children = {this_ptr};
  std::weak_ptr<Heart> result_ptr = result.heart;

  Node n(OP::OP_RELU_P, result, result.heart->children);
  State::ComputeGraph.publish(n);

  return result;
}

Tensor Tensor::Sigmoid() {
  auto this_ptr = this->heart;

  auto data = 1 / (1 + xt::exp(-this->heart->data));
  Tensor result(data);
  result.heart->children = {this_ptr};
  std::weak_ptr<Heart> result_ptr = result.heart;

  result.heart->backwards = [this_ptr, result_ptr]() {
    if (auto result_heart = result_ptr.lock()) {

      // so the derivative is i think,
      // PL/P_result (*) S';
      auto sigmoid_output = result_heart->data;

      auto upstream_grad = result_heart->gradients;
      this_ptr->gradients +=
          upstream_grad * sigmoid_output * (1.0 - sigmoid_output);
    } };

  Node n(OP::OP_SIGMOID, result, result.heart->children);
  State::ComputeGraph.publish(n);

  return result;
}

Tensor Tensor::Sigmoid_prime() {
  auto this_ptr = this->heart;

  auto S = 1 / (1 + xt::exp(-this_ptr->data));
  auto data = S * (1.0 - S);

  Tensor result(data);
  result.heart->children = {this_ptr};

  Node n(OP::OP_SIGMOID_P, result, result.heart->children);
  State::ComputeGraph.publish(n);

  return result;
}
