#include <neural/neuron.hpp>
#include <ranges>
#include <util/rand.hpp>

using namespace std;

Neuron::Neuron(size_t amount_of_inputs)
    : weights(amount_of_inputs, Node(random_float())) {}

Node Neuron::process(vector<float> inputs) {
  // this is our neuron's forward pass
  // we intake the prior layer's outputs
  // and recieve it as our inputs

  // initialise our resultant node
  Node result(0.0);
  // iterate through the inputs we got and the weights of our neuron
  // using zip just because it looks cool and we love progressive-cpp :p
  for (auto &&[input, weight] : views::zip(inputs, this->weights)) {
    // multiply the weight via the input
    // and accumulate the result
    // this is our forward pass for our neuron
    result.value += (weight.value * input);
  }

  // we then apply relu to clean up the result and pass it through to the next
  // layer.
  return result.ReLU();
}
