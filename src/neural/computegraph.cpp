#include "neural/computegraph.hpp"
#include <usul.hpp>

ComputeGraph::ComputeGraph() = default;

void ComputeGraph::publish(Node node) {
  this->nodes.emplace(node.UID, node);
}

void ComputeGraph::clear() {
  this->nodes.clear();
  State::UID = 0;
}