#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace std;

class Node : public enable_shared_from_this<Node> {
public:
  // the value of the value node
  float value;
  // the gradient of the value node
  float gradient;

  // the operation string
  string _operation;
  // the references to the children
  vector<shared_ptr<Node>> _children;

  // the backwards function as nullptr right nwo
  function<void()> _backwards;

  // the constructor MUST accept a value, the rest are optional params.
  Node(float value, string operation = "",
       vector<shared_ptr<Node>> children = {}, float gradient = 0);

  // the addition operator overload
  shared_ptr<Node> operator+(Node &other);
  // the multiplication operator overload
  shared_ptr<Node> operator*(Node &other);
  // the minus operator overload
  shared_ptr<Node> operator-(Node &other);

  shared_ptr<Node> ReLU();

  std::shared_ptr<Node> getSharedPtr() {
    // Use shared_from_this() instead of std::shared_ptr<Widget>(this)
    return shared_from_this();
  }
};
