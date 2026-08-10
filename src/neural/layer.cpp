#include <neural/layer.hpp>

Layer::Layer(int layer_size, int neuron_size) {

  neurons.reserve(layer_size);
  for (int i = 0; i < layer_size; i++) {
    neurons.emplace_back(neuron_size); // constructs a fresh, independent Neuron each time
  }
}

vector<shared_ptr<Node>> Layer::forward(vector<shared_ptr<Node>> &input) {

  vector<shared_ptr<Node>> out;

  out.reserve(this->neurons.size());

  for (Neuron neuron : this->neurons) {
    out.emplace_back(neuron.process(input));
  }

  return out;
}
