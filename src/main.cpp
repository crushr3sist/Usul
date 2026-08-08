#include <neural/node.hpp>
#include <print>

using namespace std;

int main() {
  // initial
  Node a(2.0);
  Node b(-3.0);
  Node c(10.0);
  Node f(-2.0);

  Node e(a * b);
  Node d(a + b);
  Node L(e * d);

  println("{} = {} {} {}", e.value, e._children[0].value, e._op,
          e._children[1].value);
  println("{} = {} {} {}", d.value, d._children[0].value, d._op,
          d._children[1].value);
  println("{} = {} {} {}", L.value, L._children[0].value, L._op,
          L._children[1].value);

  return 0;
}
