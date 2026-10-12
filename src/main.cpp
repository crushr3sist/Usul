#include <print>
#include <usul.hpp>

using namespace std;

int main() {

  Tensor x({{1, 2, 3},
            {4, 5, 6}});

  cout << x << "\n";

  auto y = x * 2.0;

  cout << y << "\n";
}
