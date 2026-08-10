# Usul - Neural Network & Reinforcement Learning Library

> _"The beginning of knowledge is the discovery that what you believe is not necessarily true"_
> — **Paul Muad’Dib, The Collected Sayings of Muad’Dib**

Usul is a mini neural network & reinforcement learning library, designed in Python, written in C++, created by hand for making your computer thonk a bit.

It is designed to acomplish compute graph, automatic differentiation, backpropagation and gradient descent. It wont beat your pytorch, but for simple and lightweight usecase, cant be better than Usul.

Why is it called Usul?
In Frank Herbert's Dune, the name given to Paul within the sietch was Usul. It means something close to “the base of the pillar” / “foundation” in the Dune context. The name resonated within me since Usul is the representation of what a neural network is, and how it thinks, teaching a machine to think, to be taught to believe in something, and when confronted, have faith in its beliefs. That is the reason this library is called Usul, it is the base of the pillar into something bigger.

# getting started

we're going to make our computer guess a mathematical formula, how? we're going to feed it 3 random numbers, and the computer will be able to check the out of the mathematical formula applied to those 3 random numbers, and then it will have to adjust its own weights to figure out what the formula is. So lets build this using `Usul`.

To start building your neural network, make sure you include

```cpp
#include <usul.hpp>
```

That will link the library to your project

<br>

When teaching a neural network its very important to define a learning rate, so lets define one.

```cpp
const float LR = 0.0001;
```

And remember, the smaller the better (size does matter)
<br>

Lets link 3 random numbers to our neural network. We're going to do this via creating nodes. Nodes in Usul act like parts of a map of a city, while its new, some areas of the map will be unknown to Usul, however, as Usul goes and explores, it will have a better understanding of the map. So lets introduce the first 3 points.

```cpp
auto x = Node::create(random_float());
auto y = Node::create(random_float());
auto z = Node::create(random_float());
```

X, Y and Z are the first numbers that our neural network will be exposed to. 
<br>

Now that we've got our 3 random numbers, lets plug them into a crazy maths formula
```cpp
auto target = Node::create((sin(x->value) / y->value * z->value) + 3);
```

X, Y and Z are the first numbers that Usul will be exposed to. 
<br>






