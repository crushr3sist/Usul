#include <print>
#include <usul.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

using namespace std;

xt::xarray<double> generate_random_matrix(const vector<int> &shape, double low, double high) {
  auto random_expression = xt::random::rand<double>(shape, low, high);
  xt::xarray<double> _ = random_expression;
  return _;
}

int main() {

  auto Y_true = generate_random_matrix({64, 256}, -5, 5);

  auto X = generate_random_matrix({64, 128}, -5, 5);
  auto W = generate_random_matrix({128, 256}, -5, 5);
  auto B = generate_random_matrix({1, 256}, -5, 5);

  auto Z = xt::linalg::dot(X, W);

  println("{}", Z);

  return 0;
}
