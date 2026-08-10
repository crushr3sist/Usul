#include <cstddef>
#include <math.h>
#include <neural/layer.hpp>
#include <print>
#include <util/rand.hpp>
#include <vector>

Layer::Layer(int layer_size) : neurons(layer_size, Neuron(layer_size)) {}

vector<float> Layer::forward(vector<float> input) {

  vector<float> out;

  for (Neuron neuron : this->neurons) {
    auto k = neuron.process(input).value;
    out.push_back(k);
  }

  return out;
}
