#include <print>
#include <usul.hpp>
#include <utility>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define LR 0.1

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
  auto S = Sigmoid(std::move(data));
  return S * (1.0 - S);
}

int main() {

  // we're going to now do an xor gate test

  const int EPOCH = 5000;

  xt::xarray<double> Y_true = {0.0, 1.0, 1.0, 0.0};
  Y_true.reshape({4, 1});

  xt::xarray<double> X = {
      {0.0, 0.0}, // 0.0
      {0.0, 1.0}, // 1.0
      {1.0, 0.0}, // 1.0
      {1.0, 1.0}, // 0.0
  };

  xt::xarray<double> W1 = generate_random_matrix({2, 4}, -1.0, 1.0);
  xt::xarray<double> B1 = generate_random_matrix({1, 4}, -1.0, 1.0);
  xt::xarray<double> W2 = generate_random_matrix({4, 1}, -1.0, 1.0);
  xt::xarray<double> B2 = generate_random_matrix({1, 1}, -1.0, 1.0);

  println("X shape: {}", X.shape());
  println("W_1 shape: {}", W1.shape());
  println("B_1 shape: {}", B1.shape());
  println("W_2 shape: {}", W2.shape());
  println("B_2 shape: {}", B2.shape());

  for (int i = 0; i < EPOCH; i++) {
    // *FORWARD PASS*

    // layer 1
    auto Z1 = xt::linalg::dot(X, W1) + B1;
    auto A = ReLU(Z1);
    // layer 2
    auto Z2 = xt::linalg::dot(A, W2) + B2;

    auto S = Sigmoid(Z2);

    // loss
    auto distance = S - Y_true;
    auto Loss = xt::mean(distance * distance);

    // *BACK PASS*
    double N_2 = S.shape()[0] * S.shape()[1];
    auto dLdS = (2.0 / N_2) * (S - Y_true);

    auto dLdZ_2 = dLdS * Sigmoid_prime(Z2);
    auto dLdA = xt::linalg::dot(dLdZ_2, xt::transpose(W2));
    auto dLdW_2 = xt::linalg::dot(xt::transpose(A), dLdZ_2);
    auto dLdB_2 = xt::sum(dLdZ_2, {0});

    auto dLdZ_1 = dLdA * ReLU_prime(Z1);
    auto dLdX = xt::linalg::dot(dLdZ_1, xt::transpose(W1));
    auto dLdW_1 = xt::linalg::dot(xt::transpose(X), dLdZ_1);
    auto dLdB_1 = xt::sum(dLdZ_1, {0});

    W1 = W1 - (LR * dLdW_1);
    B1 = B1 - (LR * dLdB_1);
    W2 = W2 - (LR * dLdW_2);
    B2 = B2 - (LR * dLdB_2);

    if (i % 100 == 0) {
      println("S: {}, Y_true: {}", S, Y_true);
      cout << "EPOCH: " << i << ", " << "Loss: " << Loss << '\n';
    }
  }

  return 0;
}