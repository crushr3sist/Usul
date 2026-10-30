// NOTE - once the errors go away, the library is finished :)

#include "neural/computegraph.hpp"
#include <usul.hpp>
using namespace std;

int main() {
  double LR = 0.1;
  int EPOCH = 1000;
  // lets write the forward pass

  Tensor X = {
      {0.0, 0.0}, // this is what we're feeding it
      {0.0, 1.0},
      {1.0, 0.0},
      {1.0, 1.0},
  };

  // this is what we're after
  Tensor Y_true = {
      0.0, // xor{0.0, 0.0} = 0.0
      1.0, // xor{0.0, 1.0} = 1.0
      1.0, // xor{1.0, 0.0} = 1.0
      0.0  // xor{1.0, 1.0} = 0.0
  };

  Y_true.reshape({4, 1});

  // we are now going to create a single layer, double neuron network
  // each layer will have a bias and weights tensor.
  auto W1 = generate_random_matrix({2, 4}, -1.0, 1.0);
  auto B1 = generate_random_matrix({1, 4}, -1.0, 1.0);
  auto W2 = generate_random_matrix({4, 1}, -1.0, 1.0);
  auto B2 = generate_random_matrix({1, 1}, -1.0, 1.0);

  // this is the training loop
  for (int i = 0; i < EPOCH; i++) {
    // so lets do one forward pass
    // first layer
    auto Z1 = X % W1 + B1;
    auto A = Z1.ReLU();
    // second layer
    auto Z2 = A % W2 + B2;

    // activation
    // should be - auto S = Z2.activate(Activate::Sigmoid);
    auto S = Z2.Sigmoid();
    // Loss
    // should be - auto Loss = S.loss(Loss::MSE);

    auto loss = (S - Y_true) * (S - Y_true);
    loss = loss.mean();

    cout << loss << "\n";

    loss.backwards();

    W1.step(LR);
    B1.step(LR);
    W2.step(LR);
    B2.step(LR);
  }
}
