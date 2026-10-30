#pragma once
// neural
#include <util/rand.hpp>
#include <neural/computegraph.hpp>
#include <neural/node.hpp>
#include <neural/tensor.hpp>
#include <math/activation.hpp>
#include <atomic>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/misc/xmanipulation.hpp>

namespace State {
inline constinit std::atomic<size_t> UID{0};
inline ComputeGraph ComputeGraph;

} // namespace State
