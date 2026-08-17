# getting started

we're going to make our computer guess a mathematical formula, how? we're going to feed it 3 random numbers, and the computer will be able to check the out of the mathematical formula applied to those 3 random numbers, and then it will have to adjust its own weights to figure out what the formula is. So lets build this using `Usul`.

To start building your neural network.
Make sure you include

-

```cpp
#include <usul.hpp>
```

That will link the library to your project

<br>

## When teaching a neural network

its very important to define a learning rate, so lets define one.

```cpp
const float LR = 0.0001;
```

And remember, the smaller the better (size does matter)
<br>

## Lets link 3 random numbers to our neural network.

We're going to do this via creating nodes. Nodes in Usul act like parts of a map of a city, while its new, some areas of the map will be unknown to Usul, however, as Usul goes and explores, it will have a better understanding of the map. So lets introduce the first 3 points.

```cpp
auto x = Node::create(random_float());
auto y = Node::create(random_float());
auto z = Node::create(random_float());
```

X, Y and Z are the first numbers that our neural network will be exposed to.
<br>

## let's plug them into a crazy maths formula `(sin(x) / y * z) + 3`

```cpp
auto target = Node::create((sin(x->value) / y->value * z->value) + 3);
```

X, Y and Z are our inputs. When a neural network is learning, we dont change our inputs, otherwise the network will be stuck in a loop of chasing its own tail. We have to make sure our network can atleast solve 1 problem, before its exposed to another problem.
<br>

# time to create our network
A neural network is actually a graph datastructure, if you've done linkedlist and BST's you've made a node, had to traverse a node via its children, until we find an empty space. A neural network is kind of the same. 