#include <math.h>
#include <neural/neuron.hpp>
#include <random>
#include <ranges>

using namespace std;

Neuron::Neuron(size_t amount_of_inputs) {
  // initialise weights as an empty vector
  this->weights = {};
  // loop the amount of the inputs given to this neuron and initialise at random.  
  for (size_t i = 0; i <= amount_of_inputs; i++) {
    this->weights[i] = Node(0.0);
  }
}

Node Neuron::forward(vector<float> inputs) {
  Node result(0.0);

  for (auto &&[input, weight] : views::zip(inputs, this->weights)) {
    result.value += (weight.value * input);
  }

  return result.ReLU();
}
