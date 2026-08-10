#include <memory>
#include <print>

using namespace std;

class Resource {
public:
  Resource() { println("Resource Created\n"); }
  ~Resource() { println("Resource Destroyed\n"); }
  void doSomething() { println("Did something\n"); }
};

int main() {
  // creation using make_shared
  shared_ptr<Resource> pointer1 = make_shared<Resource>();
  println("Current Owners: {}", pointer1.use_count());
  {
    // temporary existance scope
    shared_ptr<Resource> pointer2 = pointer1;
    println("Current Owners: {}", pointer1.use_count());
    pointer2->doSomething();
  }

  println("Current Owners: {}", pointer1.use_count());

  return 0;
}
