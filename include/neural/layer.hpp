#pragma once
#include <math.h>
#include <neural/neuron.hpp>
#include <vector>

using namespace std;

class Layer {

public:
  vector<Neuron> neurons;
  Layer(int layer_size, int neuron_size);
  vector<shared_ptr<Node>> forward(vector<shared_ptr<Node>> &input);
};