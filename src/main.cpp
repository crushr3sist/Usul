#include <cstdint>
#include <print>
#include <cmath>

#include <usul.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define LR 0.0001

using namespace std;

xt::xarray<double> generate_random_matrix(const vector<int> &shape, double low, double high) {
  auto random_expression = xt::random::rand<double>(shape, low, high);
  xt::xarray<double> _ = random_expression;
  return _;
}

xt::xarray<double> ReLU(xt::xarray<double> data) {
  return xt::maximum(0.0, data);
}
xt::xarray<double> ReLU_prime(xt::xarray<double> data) {
  return xt::cast<double>(data > 0.0);
}

xt::xarray<double> Sigmoid(xt::xarray<double> data) {
  return 1 / (1 + xt::exp(-data));
}
xt::xarray<double> Sigmoid_prime(xt::xarray<double> data) {
  return xt::exp(data) * (1 - xt::exp(data));
}

int main() {

  // we're going to now do an xor gate test

  const int EPOCH = 100;

  xt::xarray<double> Y_true = {0.0, 1.0, 1.0, 0.0};
  Y_true.reshape({4, 1});

  xt::xarray<double> X = {
      {0.0, 0.0}, // 0.0
      {0.0, 1.0}, // 1.0
      {1.0, 0.0}, // 1.0
      {1.0, 1.0}, // 0.0
  };

  xt::xarray<double> W1 = generate_random_matrix({2, 4}, -5, 5);
  xt::xarray<double> B1 = generate_random_matrix({1, 4}, -5, 5);

  xt::xarray<double> W2 = generate_random_matrix({4, 1}, -5, 5);
  xt::xarray<double> B2 = generate_random_matrix({4, 1}, -5, 5);

  println("X shape: {}", X.shape());
  println("W_1 shape: {}", W1.shape());
  println("B_1 shape: {}", B1.shape());
  println("W_2 shape: {}", W2.shape());
  println("B_2 shape: {}", B2.shape());

  for (int i = 0; i < EPOCH; i++) {
    // *FORWARD PASS*

    // layer 1
    auto Z1 = xt::linalg::dot(X, W1) + B1;
    println("z1 shape: {}", Z1.shape());
    auto A = ReLU(Z1);
    println("A shape: {}", A.shape());

    // layer 2
    auto Z2 = xt::linalg::dot(A, W2) + B2;
    println("Z2 shape: {}", Z2.shape());

    auto S = Sigmoid(Z2);
    println("S shape: {}", S.shape());

    println("S: {}, Y_true: {}", S, Y_true);

    println("");

    // loss
    auto distance = S - Y_true;
    println("distance shape: {}", distance.shape());
    auto Loss = xt::mean(distance * distance);
    println("loss shape: {}", Loss.shape());
    cout << "EPOCH: " << i << ", " << "Loss: " << Loss << '\n';

    // *BACK PASS*
    // REVIEW - BELOW
    // the problem is that, we're deriving A as if loss was calculated by it.
    // how do you derive A respect to L if we didnt calculate loss through it.
    // BACK PROPAGATION IS THE INVERSE OF FORWARD PASS

    // layer 2
    auto N_2 = S.shape()[0] * S.shape()[1];
    auto dL_dA_2 = 2 * (S - Y_true) / N_2;
    auto S_prime = Sigmoid_prime(Z1);
    auto dL_dZ_2 = dL_dA_2 * S_prime;
    auto dL_dB_2 = xt::sum(dL_dZ_2, 0);
    auto dL_dX_2 = xt::linalg::dot(dL_dZ_2, xt::transpose(W1));
    auto dL_dW_2 = xt::linalg::dot(xt::transpose(X), dL_dZ_2);
    W2 = W2 - (LR * dL_dW_2);
    B2 = B2 - (LR * dL_dB_2);

    

    // layer 1
    // REVIEW - BELOW
    // DERIVE pL/pA VIA THE DOT PRODUCT DERIVATIVE
    // A *  W2 + B2
    // DERIVE THAT

    auto N_1 = A.shape()[0] * A.shape()[1];
    auto dL_dA_1 = 2 * (A - Y_true) / N_1;
    auto R_prime_1 = ReLU_prime(Z1);
    auto dL_dZ_1 = dL_dA_1 * R_prime_1;
    auto dL_dB_1 = xt::sum(dL_dZ_1, 0);
    auto dL_dX_1 = xt::linalg::dot(dL_dZ_1, xt::transpose(W1));
    auto dL_dW_1 = xt::linalg::dot(xt::transpose(X), dL_dZ_1);
    W1 = W1 - (LR * dL_dW_1);
    B1 = B1 - (LR * dL_dB_1);
  }

  return 0;
}