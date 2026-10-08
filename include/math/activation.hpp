#include "neural/tensor.hpp"
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
typedef Tensor T;

T generate_random_matrix(const std::vector<int> &shape, double low, double high);
T ReLU(T data);
T ReLU_prime(T data);
T Sigmoid(T data);
T Sigmoid_prime(T data);