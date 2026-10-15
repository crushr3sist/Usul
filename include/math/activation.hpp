#include "neural/tensor.hpp"
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

Tensor generate_random_matrix(const std::vector<int> &shape, double low, double high);
Tensor ReLU(Tensor data);
Tensor ReLU_prime(Tensor data);
Tensor Sigmoid(Tensor data);
Tensor Sigmoid_prime(Tensor data);