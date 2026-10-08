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

typedef Tensor T;

T generate_random_matrix(const std::vector<int> &shape, double low, double high) {
  auto random_expression = xt::random::rand<double>(shape, low, high);
  T _(random_expression);
  return _;
}

T ReLU(T data) {
  data.data = xt::maximum(0.0, data.data);
  // we still need to add it to the compute graph
  vector<Tensor *> t = {&data};
  Node n(OP::OP_RELU, false, &data, t);
  State::ComputeGraph.publish(n);

  return data;
}

T ReLU_prime(T data) {
  data.data = xt::cast<double>(data.data > 0.0);

  vector<Tensor *> t = {&data};
  Node n(OP::OP_RELU_P, false, &data, t);
  State::ComputeGraph.publish(n);

  return data;
}

T Sigmoid(T data) {
  data.data = 1 / (1 + xt::exp(-data.data));

  vector<Tensor *> t = {&data};
  Node n(OP::OP_SIGMOID, false, &data, t);
  State::ComputeGraph.publish(n);

  return data;
}

T Sigmoid_prime(T data) {
  auto S = Sigmoid(data.data);
  data.data = S.data * (1.0 - S.data);

  vector<Tensor *> t = {&data};
  Node n(OP::OP_SIGMOID_P, false, &data, t);
  State::ComputeGraph.publish(n);

  return data;
}
