
#include <math.h>
#include <memory>
#include <neural/node.hpp>
#include <vector>

using namespace std;

Node::Node(float value, string operation, vector<shared_ptr<Node>> children,
           float gradient) {

  this->value = value;
  this->_operation = operation;
  this->_children = children;
  this->gradient = gradient;
  this->_backwards = []() {};
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
 * @note Performance is O(1) time complexity.
 */
shared_ptr<Node> Node::operator+(Node &other) {
  // generate a list of children
  vector<shared_ptr<Node>> children;
  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  // generate the resultant value instance
  shared_ptr<Node> result = make_shared<Node>(this->value + other.value, "+", children);
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
 * @note Performance is O(1) time complexity.
 */
shared_ptr<Node> Node::operator*(Node &other) {
  // generate a list of children
  vector<shared_ptr<Node>> children;
  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  // generate the resultant value instance
  shared_ptr<Node> result = make_shared<Node>(this->value * other.value, "*", children);

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
 * @note Performance is O(1) time complexity.
 */
shared_ptr<Node> Node::operator-(Node &other) {
  // generate a list of children
  vector<shared_ptr<Node>> children;

  auto this_ptr = this->getSharedPtr();
  auto other_ptr = other.getSharedPtr();

  children.emplace_back(this_ptr);
  children.emplace_back(other_ptr);

  // generate the resultant value instance
  shared_ptr<Node> result = make_shared<Node>(this->value - other.value, "-", children);
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
 * @note Performance is O(1) time complexity.
 */
shared_ptr<Node> Node::ReLU() {
  vector<shared_ptr<Node>> children;
  auto this_ptr = this->getSharedPtr();
  children.emplace_back(this_ptr);

  shared_ptr<Node> result = make_shared<Node>(fmax(0.0F, this->value), "ReLU", children);
  auto *result_ptr = result.get();

  result->_backwards = [this_ptr, result_ptr]() {
    // if the value of our node is more than 0, we'll continue
    if (this_ptr->value > 0.0F) {
      this_ptr->gradient += float(1.0) * result_ptr->gradient;
    }
    // if its not, then 0 multiplied by anything wil be 0, so there's no point computing that
  };

  return result;
};
