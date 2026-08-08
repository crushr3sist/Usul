#include "node.hpp"
#include <vector>

using namespace std;

class Neuron {
public:
  Node bias{0.0};
  vector<Node> weights;
  Neuron(size_t amount_of_inputs);
  Node process(vector<float> input);
};