#include <usul.hpp>
#include <print>

using namespace std;

int main() {

  const float LR = 0.0001;

  auto x = Node::create(random_float());
  auto y = Node::create(random_float());
  auto z = Node::create(random_float());

  auto target = Node::create((sin(x->value) / y->value * z->value) + 3);

  vector<Layer> network;

  Layer hidden_layer1(3, 3);
  Layer hidden_layer2(3, 3);
  Layer hidden_layer3(3, 3);
  Layer output(1, 3);

  network.push_back(hidden_layer1);
  network.push_back(hidden_layer2);
  network.push_back(hidden_layer3);
  network.push_back(output);

  for (size_t i = 0; i <= 50; i++) {

    vector input_vector = {x, y, z};

    auto hidden_layer1_output = network[0].forward(input_vector);
    auto hidden_layer2_output = network[1].forward(hidden_layer1_output);
    auto hidden_layer3_output = network[2].forward(hidden_layer2_output);
    auto prediction = network[3].forward(hidden_layer3_output)[0];

    println("EPOCH {}", i);
    println();
    println("prediction: {}", prediction->value);
    println("target: {}", target->value);
    auto error = (*prediction - *target);

    auto loss = *(*prediction - *target) * *(*prediction - *target);

    println("Loss: {}", loss->value);

    autodiff(loss);

    gradient_descent(network, LR);
  }

  return 0;
}
