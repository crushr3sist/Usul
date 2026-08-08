#include <functional>
#include <math.h>
#include <neural/node.hpp>
#include <vector>

Node::Node(float value, std::string op, std::vector<Node> children,
           float gradient) {
  this->value = value;
  this->_op = op;
  this->_children = children;
  this->gradient = gradient;
  this->_backwards = []() {};
}

Node Node::operator+(Node &other) {
  // generate a list of children
  std::vector<Node> children = {*this, other};

  // generate the resultant value instance
  auto result = new Node(this->value + other.value, _op = "+", children);

  // define the node's back-prop
  result->_backwards = [this, &result, &other]() {
    // do the backwards propagation
    this->gradient += 1 * result->gradient;
    other.gradient += 1 * result->gradient;
  };

  return *result;
}

Node Node::operator*(Node &other) {
  // generate a list of children
  std::vector<Node> children = {*this, other};

  // generate the resultant value instance
  auto result = new Node(this->value * other.value, _op = "*", children);

  // define the node's back-prop
  result->_backwards = [this, &result, &other]() {
    // do the backwards propagation
    this->gradient += other.value * result->gradient;
    other.gradient += this->value * result->gradient;
  };

  return *result;
}

Node Node::operator-(Node &other) {
  // generate a list of children
  std::vector<Node> children = {*this, other};

  // generate the resultant value instance
  auto result = new Node(this->value - other.value, _op = "-", children);

  // define the node's back-prop
  result->_backwards = [this, &result, &other]() {
    // do the backwards propagation
    this->gradient += 1 * result->gradient;
    other.gradient += -1 * result->gradient;
  };
  return *result;
}

Node Node::ReLU() {
  std::vector<Node> children = {*this};
  auto result = new Node(fmax(0, int(this->value)), _op = "ReLU", children);

  result->_backwards = [this, &result]() {
    this->gradient += [this]() {
      if (this->value >= 0) {
        return float(1.0);
      } else {
        return float(0.0);
      }
    }();
  };

  return *result;
};
