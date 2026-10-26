#include "neural/tensor.hpp"
#include <usul.hpp>

using namespace std;

int main() {

  /*
  now all thats left is auto diffing. the backwards propagation of error is going to be okay, but we need from node
  backwards propagation of error.

  so each operation we have, it makes a node, that node is a simple node in the compute graph.
  the compute graph is a 1-dimensional linear graph, its not acyclic, its kinda just a linked list at this point
  so following the nodes. lets check to see what happens, when we provide a complex base of operations to occur.
  */

  const int EPOCH = 5000;
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
    auto Z1 = X % W1 + B1;
  }
}
