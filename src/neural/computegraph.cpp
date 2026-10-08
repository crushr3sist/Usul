#include <neural/computegraph.hpp>

ComputeGraph::ComputeGraph() = default;

void ComputeGraph::publish(Node node) {
  this->nodes.emplace(node.UID, node);
}
