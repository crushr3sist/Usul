#pragma once

#include "neural/node.hpp"
#include <unordered_map>
class ComputeGraph {
  // this is the class that keeps track of our
public:
  void publish(Node node);
  ComputeGraph();
  std::unordered_map<uint8_t, Node> nodes;
};

