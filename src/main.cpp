#include "neural/tensor.hpp"
#include <usul.hpp>
#include <typeinfo>

using namespace std;

int main() {

  Tensor x({{1, 2, 3},
            {4, 5, 6}});

  cout << x << "\n";

  auto lhs_plus = 1.0 + x;
  auto rhs_plus = x + 1.0;
  auto t2t_plus = x + x;

  println("{}", typeid(lhs_plus) == typeid(Tensor));
  println("{}", typeid(rhs_plus) == typeid(Tensor));
  println("{}", typeid(t2t_plus) == typeid(Tensor));

  auto lhs_mul = 1.0 + x;
  auto rhs_mul = x + 1.0;
  auto t2t_mul = x + x;

  println("{}", typeid(lhs_mul) == typeid(Tensor));
  println("{}", typeid(rhs_mul) == typeid(Tensor));
  println("{}", typeid(t2t_mul) == typeid(Tensor));

  auto lhs_sub = 1.0 - x;
  auto rhs_sub = x - 1.0;
  auto t2t_sub = x - x;

  println("{}", typeid(lhs_mul) == typeid(Tensor));
  println("{}", typeid(rhs_mul) == typeid(Tensor));
  println("{}", typeid(t2t_mul) == typeid(Tensor));

  auto k = x + -x;
  cout << k << "\n";
}
