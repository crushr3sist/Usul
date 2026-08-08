#include <algorithm>
#include <functional>
#include <math.h>
#include <print>
#include <vector>
using namespace std;

class Layer {
public:
  Layer(int layer_size);
  vector<float> forward(vector<float> input);
};