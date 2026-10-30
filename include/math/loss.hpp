#include "neural/tensor.hpp"
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

Tensor MSE();
Tensor MSE_Prime();

Tensor BGD();
Tensor BGD_prime();
Tensor SGD();
Tensor SGD_prime();
