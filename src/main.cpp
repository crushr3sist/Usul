#include <usul.hpp>
using namespace std;

int main() {
  Tensor A = {1.0, 2.0, 3.0};
  Tensor B = {4.0, 5.0, 6.0};
  auto Y = A + B;

  cout << Y << "\n";
}
