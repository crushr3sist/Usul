#include "neural/tensor.hpp"
#include <usul.hpp>
#include <typeinfo>

using namespace std;

int main() {

  Tensor x({{1, -2, -3},
            {4, 5, 6}});

  cout << x << "\n";

  auto k = x.ReLU();
  cout << k;
}
