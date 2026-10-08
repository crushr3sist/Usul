#include <print>
#include <xtensor/core/xmath.hpp>
#include "neural/tensor.hpp"
#include "usul.hpp"
#include <utility>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor.hpp>
#include <xtensor/core/xshape.hpp>

// #define LR 0.1

using namespace std;

int main() {

  println("{}", (int)State::UID);

  Tensor A(xt::zeros<double>({3, 4}));
  Tensor B(xt::zeros<double>({1, 4}));

  auto Y = A + B;
  println("{}", Y.data);
}
