#pragma once
// neural
#include "neural/computegraph.hpp"
#include <neural/node.hpp>
#include <util/rand.hpp>
#include <atomic>

namespace State {
inline constinit std::atomic<uint8_t> UID{0};
inline ComputeGraph ComputeGraph;

} // namespace State