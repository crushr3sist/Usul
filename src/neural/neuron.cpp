#include "neural/node.hpp"
#include <neural/neuron.hpp>
#include <print>
#include <ranges>
#include <util/rand.hpp>

using namespace std;

Neuron::Neuron(size_t amount_of_inputs) {
  weights.reserve(amount_of_inputs);

  for (size_t i = 0; i < amount_of_inputs; i++) {
    weights.emplace_back(Node::create(random_float() * 0.1F));
  }

  bias = Node::create(random_float() * 0.1F);
}

shared_ptr<Node> Neuron::process(vector<shared_ptr<Node>> inputs) {
  // this is our neuron's forward pass
  // we intake the prior layer's outputs
  // and recieve it as our inputs

  // initialise our resultant node
  auto result = Node::create(0.0);
  // iterate through the inputs we got and the weights of our neuron
  // using zip just because it looks cool and we love progressive-cpp :p

  for (auto const &[input, weight] : std::views::zip(inputs, this->weights)) {
    result = *result + *(*weight * *input);
  }

  // finally add the bias from our neuron.
  result = *result + *this->bias;

  // we then apply relu to clean up the result and pass it through to the next
  // layer.
  return result->ReLU();
}
