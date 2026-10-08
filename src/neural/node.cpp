#include "usul.hpp"
#include <neural/node.hpp>

Node::Node(uint8_t op, bool broadcasted, Tensor *tensor_ptr, vector<Tensor *> parents) {
  this->op = op;
  this->tensor_ptr = tensor_ptr;
  this->parents = parents;
  this->broadcasted = broadcasted;
  this->UID = increment_uid(); //

}

uint8_t Node::increment_uid() {
  // check what the number is, and plus 1 to it.
  State::UID += 1;
  return State::UID;
}