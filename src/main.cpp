#include <cstdint>
#include <print>
#include <usul.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define LR 0.01

using namespace std;

xt::xarray<double> generate_random_matrix(const vector<int> &shape, double low, double high) {
  auto random_expression = xt::random::rand<double>(shape, low, high);
  xt::xarray<double> _ = random_expression;
  return _;
}

xt::xarray<double> ReLU(xt::xarray<double> data) {
  return xt::maximum(0, data);
}

xt::xarray<double> ReLU_prime(xt::xarray<double> data) {
  return xt::cast<double>(data > 0.0);
}

int main() {

  const int EPOCH = 1000;

  // this is our truth matrix
  auto Y_true = generate_random_matrix({64, 256}, -5, 5);

  // this is our inputs matrix
  auto X = generate_random_matrix({64, 128}, -5, 5);

  // this is our weights matrix
  auto W = generate_random_matrix({128, 256}, -5, 5);

  // and this is our bias matrix
  auto B = generate_random_matrix({1, 256}, -5, 5);

  for (int i = 0; i < EPOCH; i++) {
    // *FORWARD PASS*
    // this is our forward pass
    auto Z = xt::linalg::dot(X, W) + B;
    // ReLU activation
    auto A = ReLU(Z);
    // LOSS mse
    auto distance = A - Y_true;
    auto Loss = xt::mean(distance * distance);
    cout << "EPOCH: " << i << ", " << "Loss: " << Loss << '\n';

    // *BACKWARDS*
    // we're using MSE for loss

    auto N = A.shape()[0] * A.shape()[1];
    auto dL_dA = 2 * (A - Y_true) / N;

    auto R_prime = ReLU_prime(Z);
    auto dL_dZ = dL_dA * R_prime;
    auto dL_dB = xt::sum(dL_dZ, 0);
    auto dL_dX = xt::linalg::dot(dL_dZ, xt::transpose(W));

    auto dL_dW = xt::linalg::dot(xt::transpose(X), dL_dZ);

    // gradient descent
    W = W - (LR * dL_dW);
    B = B - (LR * dL_dB);
  }

  return 0;
}