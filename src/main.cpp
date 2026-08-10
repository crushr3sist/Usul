#include <cmath>
#include <cstddef>
#include <neural/layer.hpp>
#include <optim/autodiff.hpp>
#include <optim/gradient_descent.hpp>
#include <print>
#include <util/rand.hpp>

using namespace std;

int main() {
  const float LR = 0.001;

  Node x(random_float());
  Node y(random_float());
  Node z(random_float());

  Node target((sin(x.value) / y.value * z.value) + 3);

  vector<Layer> network;

  Layer hidden_layer1(3);
  Layer hidden_layer2(3);
  Layer hidden_layer3(3);
  Layer output(1);

  network.emplace_back(hidden_layer1);
  network.emplace_back(hidden_layer2);
  network.emplace_back(hidden_layer3);
  network.emplace_back(output);

  for (size_t i = 0; i <= 50; i++) {

    auto hidden_layer1_output = network[0].forward({x.value, y.value, z.value});
    auto hidden_layer2_output = network[1].forward(hidden_layer1_output);
    auto hidden_layer3_output = network[2].forward(hidden_layer2_output);
    auto prediction = network[3].forward(hidden_layer3_output)[0];

    println("EPOCH {}", i);
    println();
    println("prediction: {}", prediction);
    println("target: {}", target.value);

    Node loss((prediction - target.value) * (prediction - target.value));

    println("Loss: {}", loss.value);

    autodiff(loss);

    gradient_descent(network, LR);
  }

  return 0;
}
