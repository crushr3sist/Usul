#include <usul/usul.hpp>

int main() {
  double LR = 0.1;
  /*
  now all thats left is auto diffing. the backwards propagation of error is going to be okay, but we need from node
  backwards propagation of error.

  so each operation we have, it makes a node, that node is a simple node in the compute graph.
  the compute graph is a 1-dimensional linear graph, its not acyclic, its kinda just a linked list at this point
  so following the nodes. lets check to see what happens, when we provide a complex base of operations to occur.
  */

  const int EPOCH = 1000;
  Tensor Y_true = {0.0, 1.0, 1.0, 0.0};
  Y_true.reshape({4, 1});

  Tensor X = {
      {0.0, 0.0}, // 0.0
      {0.0, 1.0}, // 1.0
      {1.0, 0.0}, // 1.0
      {1.0, 1.0}, // 0.0
  };

  auto W1 = generate_random_matrix({2, 4}, -1.0, 1.0);
  auto B1 = generate_random_matrix({1, 4}, -1.0, 1.0);
  auto W2 = generate_random_matrix({4, 1}, -1.0, 1.0);
  auto B2 = generate_random_matrix({1, 1}, -1.0, 1.0);

  println("X shape: {}", X.shape());
  println("W_1 shape: {}", W1.shape());
  println("B_1 shape: {}", B1.shape());
  println("W_2 shape: {}", W2.shape());
  println("B_2 shape: {}", B2.shape());

  for (int i = 0; i < EPOCH; i++) {
    // layer 1
    auto Z1 = X % W1 + B1;
    auto A = Z1.ReLU();
    // layer 2
    auto Z2 = A % W2 + B2;

    // activation
    auto S = Z2.Sigmoid();

    // loss
    auto distance = S - Y_true;
    auto Loss = (distance * distance).mean();

    // backpass
    auto N_2 = S.shape()[0] * S.shape()[1];
    auto dLdS = (2.0 / N_2) * (S - Y_true);

    auto dLdZ_2 = dLdS * Z2.Sigmoid_prime();
    auto dLdA = dLdZ_2 % W2.transpose();
    auto dLdW_2 = A.transpose() % dLdZ_2;
    auto dLdB_2 = dLdZ_2.sum({0});

    auto dLdZ_1 = dLdA * Z1.ReLU_prime();
    auto dLdX = dLdZ_1 % W1.transpose();
    auto dLdW_1 = X.transpose() % dLdZ_1;
    auto dLdB_1 = dLdZ_1.sum({0});

    W1 = W1 - (LR * dLdW_1);
    B1 = B1 - (LR * dLdB_1);
    W2 = W2 - (LR * dLdW_2);
    B2 = B2 - (LR * dLdB_2);
    std::cout << "S: " << S << ", Y_true: " << Y_true << "\n";
    std::cout << "EPOCH: " << i << ", " << "Loss: " << Loss << '\n';
  }
}