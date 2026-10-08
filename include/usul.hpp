#pragma once
// neural
#include <neural/computegraph.hpp>
#include <neural/node.hpp>
#include <neural/tensor.hpp>
#include <math/activation.hpp>
#include <atomic>

namespace State {
inline constinit std::atomic<uint8_t> UID{0};
inline ComputeGraph ComputeGraph;

} // namespace State