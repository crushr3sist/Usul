#include <algorithm>
#include <optim/autodiff.hpp>
#include <unordered_set>

using namespace std;

void autodiff(Node &loss) {
  // this is our handy dandy little autodiff
  //
  // when a node does a mathematical operation with another node, the resultant
  // node is supplied with thier chain rule expression. our autodiffer sorts the
  // children nodes, prefereably from the loss node using topological sort with
  // DFS and then calls the back-propagation functions inside of each non-leaf
  // nodes.

  // our comp graph container
  vector<Node> nodes;
  // our visited set
  unordered_set<Node> visted;

  // a little awesome lambda function
  // to use DFS to create a flattened list of our nodes in the compute graph.
  auto topological_sort = [&visted](this auto self, Node v) -> void {
    // checking the the value being looked at is not in the visited set
    if (!visted.contains(v)) {
      // we insert the node into visited
      visted.insert(v);
      // continue to iterate through thier children
      for (const auto &child : v._children) {
        // and sort thier children and so forth
        self(child);
      }
    }
    // until we dont have any more children left
    // which is where we exit.
    return;
  };

  // execute the topological sort on our loss node
  topological_sort(loss);
  // initialise the loss node to 1.0 as dL/dL = 1
  loss.gradient = 1.0;
  // reverse our flattened list of nodes
  ranges::reverse(nodes);

  // and finally iterate through our list of nodes
  for (const Node &node : nodes) {
    // and propagate the error backwards in the compute graph.
    // boom
    node._backwards();
  }
}
