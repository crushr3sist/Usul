#include "neural/tensor.hpp"
#include <usul.hpp>

using namespace std;

int main() {
  // 1. make a shared ptr first
  // we need to create a tensor, and then do some equations
  // and then be able to call backward() for the last node in the equation

  Tensor k = {{1.0, 2.0, 3.0}};
  cout << k << "\n";

  // 

}
