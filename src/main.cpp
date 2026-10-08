#include <print>
#include <usul.hpp>

using namespace std;

int main() {

  const int EPOCH = 5000;

  // the truth values
  Tensor Y_true({0.0, 1.0, 1.0, 0.0});
  Y_true.data.reshape({4, 1});

  // our data
  Tensor X(
      {
          {0.0, 0.0},
          {0.0, 1.0},
          {1.0, 0.0},
          {1.0, 1.0},

      });

  Tensor W1 = generate_random_matrix({2, 4}, -1.0, 1.0);
  Tensor B1 = generate_random_matrix({1, 4}, -1.0, 1.0);
  Tensor W2 = generate_random_matrix({4, 1}, -1.0, 1.0);
  Tensor B2 = generate_random_matrix({1, 1}, -1.0, 1.0);
  println("{}", (int)State::UID);
  auto Z1 = X ^ W1 + B1;
  println("{}", (int)State::UID);
  auto A = ReLU(Z1);
  println("{}", (int)State::UID);
  auto Z2 = A ^ W2 + B2;
  println("{}", (int)State::UID);
  auto S = Sigmoid(Z2);
  println("{}", (int)State::UID);
}
