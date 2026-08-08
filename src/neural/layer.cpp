#include <cstddef>
#include <math.h>
#include <neural/layer.hpp>
#include <util/rand.hpp>
#include <vector>

Layer::Layer(int layer_size) {
  this->neurons = {};

  for (size_t i = 0; i <= layer_size; i++) {
    this->neurons[i] = Neuron(layer_size);
  }
}
vector<Node> Layer::forward(vector<float> input) {

  vector<Node> out;

  for (Neuron neuron : this->neurons) {
    auto k = neuron.process(input);
    out.push_back(k);
  }

  return out;
}
