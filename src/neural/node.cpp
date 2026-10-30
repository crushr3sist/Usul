#include "usul.hpp"

Node::Node(
    uint8_t op,
    Tensor result_tensor,
    std::vector<std::shared_ptr<Heart>> parents)
    : UID(increment_uid()),
      op(op),
      resultant_tensor(std::move(result_tensor)),
      parents(std::move(parents)) {
}

size_t Node::increment_uid() {
  // check what the number is, and plus 1 to it.
  State::UID.fetch_add(1);
  return State::UID;
}