#pragma once

// our class node is the format for a tensor
// so that we can include it into the compute graph.

#include "neural/tensor.hpp"
#include <cstdint>

using namespace std;

enum OP : uint8_t {
  OP_ADD = 1,      // + addition
  OP_HADAMARD = 2, // * element-wise
  OP_DOT = 3,      // ^ dot product
  OP_SUB = 4,      // ^ dot product

  // activation
  OP_RELU = 10,
  OP_RELU_P = 11,
  OP_SIGMOID = 12,
  OP_SIGMOID_P = 13,

  // loss
  OP_MEAN = 20

};
class Node {
  // this is what is going to wrap our tensor.
  // our tensor is the data,
  // Node is our government id

public:
  Node(
      uint8_t op,
      Tensor result_tensor,
      std::vector<std::shared_ptr<Heart>> parents);

  size_t UID;
  uint8_t op;

  Tensor resultant_tensor;
  std::vector<std::shared_ptr<Heart>> parents;

  static size_t increment_uid();
};