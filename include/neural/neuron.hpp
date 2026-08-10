#pragma once

#include "node.hpp"
#include <memory>
#include <vector>

using namespace std;

class Neuron {
public:
  shared_ptr<Node> bias;
  vector<shared_ptr<Node>> weights;
  Neuron(size_t amount_of_inputs);
  shared_ptr<Node> process(vector<shared_ptr<Node>> input);
};