#include <neural/computegraph.hpp>

ComputeGraph::ComputeGraph() = default;

void ComputeGraph::publish(Node node) {
  this->nodes.emplace(node.UID, node);
}

void ComputeGraph::backwards() {

  for (size_t i = this->nodes.size(); i > 0; i--) {
    println("uid: {}", this->nodes.at(i).UID);
    println("op: {}", this->nodes.at(i).op);
  }
}