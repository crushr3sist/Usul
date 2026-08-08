#include <math.h>
#include <neural/neuron.hpp>
#include <vector>

using namespace std;

class Layer {

public:
  vector<Neuron> neurons;
  Layer(int layer_size);
  vector<Node> forward(vector<float> input);
};