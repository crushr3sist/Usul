#include <optim/gradient_descent.hpp>

// 1. Add '&' to pass the original network, not a copy
void gradient_descent(std::vector<Layer> &Layers, float learning_rate) {

  // 2. Add '&' to grab references to the layers, neurons, and weights
  for (auto &layer : Layers) {

    for (auto &neuron : layer.neurons) {

      neuron.bias.value -= learning_rate * neuron.bias.gradient;

      for (auto &weight : neuron.weights) {
        weight.value -= learning_rate * weight.gradient;
      }

      neuron.bias.gradient = 0.0;

      for (auto &weight : neuron.weights) {
        weight.gradient = 0.0;
      }
    }
  }
}