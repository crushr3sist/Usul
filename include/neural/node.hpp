#include <functional>
#include <string>
#include <vector>

class Node {
public:
  // the value of the value node
  float value;
  // the gradient of the value node
  float gradient;

  // the operation string
  std::string _operation;
  // the references to the children
  std::vector<Node> _children;

  // the backwards function as nullptr right nwo
  std::function<void()> _backwards;

  // the constructor MUST accept a value, the rest are optional params.
  Node(float value, std::string operation = "", std::vector<Node> children = {},
       float gradient = 0);

  // the addition operator overload
  Node operator+(Node &other);
  // the multiplication operator overload
  Node operator*(Node &other);
  // the minus operator overload
  Node operator-(Node &other);

  Node ReLU();
};