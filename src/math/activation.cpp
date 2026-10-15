#include "neural/node.hpp"
#include "usul.hpp"
#include <neural/tensor.hpp>
#include <math/activation.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

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

  auto data = xt::maximum(0.0, this->data);
  Tensor t(data);
  // we still need to add it to the compute graph
  vector<Tensor *> p = {this};
  Node n(OP::OP_RELU, false, &t, p);
  State::ComputeGraph.publish(n);

  return t;
}

Tensor Tensor::ReLU_prime() {
  auto data = xt::cast<double>(this->data > 0.0);
  Tensor t(data);

  vector<Tensor *> p = {this};
  Node n(OP::OP_RELU_P, false, &t, p);
  State::ComputeGraph.publish(n);

  return t;
}

Tensor Tensor::Sigmoid() {
  auto data = 1 / (1 + xt::exp(-this->data));
  Tensor t(data);

  vector<Tensor *> p = {this};
  Node n(OP::OP_SIGMOID, false, &t, p);
  State::ComputeGraph.publish(n);

  return t;
}

Tensor Tensor::Sigmoid_prime() {
  auto S = Sigmoid();
  auto data = 1 / (1 + xt::exp(-this->data));

  auto results = S.data * (1.0 - S.data);
  Tensor t(results);
  vector<Tensor *> p = {this};
  Node n(OP::OP_SIGMOID_P, false, &t, p);
  State::ComputeGraph.publish(n);

  return t;
}
