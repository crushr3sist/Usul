#include "usul.hpp"
#include <neural/node.hpp>
#include <utility>

Node::Node(
    uint8_t op,
    bool broadcasted,
    Tensor result_tensor,
    std::vector<std::shared_ptr<Heart>> parents)
    : UID(increment_uid()),
      op(op),
      broadcasted(broadcasted),
      resultant_tensor(std::move(result_tensor)),
      parents(std::move(parents)) {
}

uint8_t Node::increment_uid() {
  // check what the number is, and plus 1 to it.
  State::UID += 1;
  return State::UID;
}