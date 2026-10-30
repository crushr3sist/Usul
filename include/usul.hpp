#pragma once
// neural

#include <neural/computegraph.hpp>
#include <neural/node.hpp>
#include <neural/tensor.hpp>
#include <math/activation.hpp>
#include <atomic>
#include <xtensor/io/xio.hpp>

namespace State {
inline constinit std::atomic<size_t> UID{0};
inline ComputeGraph ComputeGraph;

} // namespace State
