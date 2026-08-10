
#include <math.h>
#include <memory>
#include <neural/node.hpp>
#include <utility>
#include <vector>

using namespace std;

/**
 * @brief constructor function for Node
 *
 * to improve the DX for creating nodes, a simple constructor is created to avoid having
 * developers write shared_ptr<node> everytime they want to create a new node.
 *
 * @param[in] float value the value of the node.
 * @param[in] vector<shared_ptr<Node>> the children of the new parent node .
 * @param[in] float gradient the gradient of the current parent node.
 * @return shared pointer of the current node instance.
 *
 * @throws nothing.
 *
 */
Node::Node(float value, vector<shared_ptr<Node>> children, float gradient) {

  this->value = value;
  this->_children = std::move(children);
  this->gradient = gradient;
  this->_backwards = []() {};
}

/**
 * @brief creator function for Node
 *
 * to improve the DX for creating nodes, a simple constructor is created to avoid having
 * developers write shared_ptr<node> everytime they want to create a new node.
 *
 * @param[in] float value the value of the node.
 * @param[in] vector<shared_ptr<Node>> the children of the new parent node .
 * @param[in] float gradient the gradient of the current parent node.
 * @return shared pointer of the current node instance.
 *
 * @throws nothing.
 *
 */
shared_ptr<Node> Node::create(float value, vector<shared_ptr<Node>> children, float gradient) {
  // we're moving the operation and children, as we do not need thier copies, they can belong to this node's instance
  shared_ptr<Node> new_node = shared_ptr<Node>(new Node(value, std::move(children), gradient));

  return new_node;
}

/**
 * @brief addition operator overload for Node
 *
 * Overloads the addition operator for the Node class.
 * Node to Node addition results in new parent node addition in computation graph
 * Handled via shared pointers and closure generation for backward's propagation of errors.
 *
 * @param[in] Node &other the leaf node or RHS of the equation.
 * @return The new parent node.
 *
 * @throws nothing.
 *
 */
shared_ptr<Node> Node::operator+(Node &other) {
  // generate a list of children
  vector<shared_ptr<Node>> children;

  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  // generate the resultant value instance
  shared_ptr<Node> result = Node::create(this->value + other.value, children);

  auto *result_ptr = result.get();

  // define the node's back-prop
  result->_backwards = [this_ptr, other_ptr, result_ptr]() {
    // do the backwards propagation
    this_ptr->gradient += 1 * result_ptr->gradient;
    other_ptr->gradient += 1 * result_ptr->gradient;
  };

  return result;
}

/**
 * @brief multiplication operator overload for Node
 *
 * Overloads the multiplication operator for the Node class.
 * Node to Node multiplication results in new parent node multiplication in computation graph
 * Handled via shared pointers and closure generation for backward's propagation of errors.
 *
 * @param[in] Node &other the leaf node or RHS of the equation.
 * @return The new parent node.
 *
 * @throws nothing.
 *
 */
shared_ptr<Node> Node::operator*(Node &other) {
  // generate a list of children
  vector<shared_ptr<Node>> children;
  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  // generate the resultant value instance
  shared_ptr<Node> result = Node::create(this->value * other.value, children);

  auto *result_ptr = result.get();

  // define the node's back-prop
  result->_backwards = [this_ptr, other_ptr, result_ptr]() {
    // do the backwards propagation
    this_ptr->gradient += other_ptr->value * result_ptr->gradient;
    other_ptr->gradient += this_ptr->value * result_ptr->gradient;
  };

  return result;
}

/**
 * @brief subtraction operator overload for Node
 *
 * Overloads the subtraction operator for the Node class.
 * Node to Node subtraction results in new parent node subtraction in computation graph
 * Handled via shared pointers and closure generation for backward's propagation of errors.
 *
 * @param[in] Node &other the leaf node or RHS of the equation.
 * @return The new parent node.
 *
 * @throws nothing.
 *
 */
shared_ptr<Node> Node::operator-(Node &other) {
  // generate a list of children
  vector<shared_ptr<Node>> children;

  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  // generate the resultant value instance
  shared_ptr<Node> result = Node::create(this->value - other.value, children);
  auto *result_ptr = result.get();

  // define the node's back-prop
  result->_backwards = [this_ptr, other_ptr, result_ptr]() {
    // do the backwards propagation
    this_ptr->gradient += 1 * result_ptr->gradient;
    other_ptr->gradient += -1 * result_ptr->gradient;
  };

  return result;
}

/**
 * @brief Rectified Linear Unit applicator for Node
 *
 * applies ReLU function for the Node class.
 * Node to Node ReLU results in new parent node ReLU in computation graph
 * Handled via shared pointers and closure generation for backward's propagation of errors.
 *
 * @return The new parent node with value cleaned via ReLU.
 *
 * @throws nothing.
 *
 */
shared_ptr<Node> Node::ReLU() {
  vector<shared_ptr<Node>> children;
  auto this_ptr = this->getSharedPtr();
  children.emplace_back(this_ptr);

  shared_ptr<Node> result = Node::create(fmax(0.0F, this->value), children);
  auto *result_ptr = result.get();

  result->_backwards = [this_ptr, result_ptr]() {
    // if the value of our node is more than 0, we'll continue
    if (this_ptr->value > 0.0F) {
      this_ptr->gradient += float(1.0) * result_ptr->gradient;
    } else {
      this_ptr->gradient += 0.0F * result_ptr->gradient;
    }
  };

  return result;
};
