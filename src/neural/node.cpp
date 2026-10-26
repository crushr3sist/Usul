#include "usul.hpp"
#include <neural/node.hpp>

Node::Node(uint8_t op, bool broadcasted, Tensor result_tensor, std::vector<std::shared_ptr<Heart>> parents) {
  this->op = op;
  this->tensor_ptr = result_tensor;
  this->parents = parents;
  this->broadcasted = broadcasted;
  this->UID = increment_uid(); //
}

uint8_t Node::increment_uid() {
  // check what the number is, and plus 1 to it.
  State::UID += 1;
  return State::UID;
}