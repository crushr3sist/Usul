#pragma once

#include "neural/node.hpp"
#include <unordered_map>
class ComputeGraph {
  // this is the class that keeps track of our
public:
  void publish(Node node);
  ComputeGraph();
  std::unordered_map<size_t, Node> nodes;
  void backwards();
};
