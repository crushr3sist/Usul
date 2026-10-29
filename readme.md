# Usul

Usul is a Deep Neural Network and automatic differentiation library written in modern C++23 on top of `xtensor`. I wrote this library for people who want to express their models directly as mathematics in C++, without fighting the massive boilerplate and systems overhead of LibTorch or TensorFlow's C++ APIs.

The way it works is simple: just write your tensor maths. Your tensors support native mathematical operators while Usul tracks the computation history under the hood for backpropagation.

For example:

```cpp
Tensor A = {1.0, 2.0, 3.0};
Tensor B = {4.0, 5.0, 6.0};

auto Y = A + B;
std::cout << Y << '\n';
```

Output:

```text
{ 5., 7., 9. }, shape=(3)
```

## How does it work?

The core class you interact with is `Tensor`, which acts as a lightweight value-type handle wrapping a `std::shared_ptr<Heart>`.

The actual tensor data and gradient state inside `Heart` are decoupled from the `Tensor` handle itself. Copying or passing a `Tensor` around is cheap, and the backend mixes `std::shared_ptr` (for parent tracking) and `std::weak_ptr` (inside backward pass closures) to propagate gradients efficiently without cyclic memory leaks.

Usul uses operator overloading so matrix mathematics read like standard equations. As an expression evaluates at runtime, each operation automatically registers a node and its local derivative closure to the global compute graph:

```cpp
Tensor A = {{1.0, 2.0}, {3.0, 4.0}};
Tensor B = {{4.0, 1.0}, {2.0, 2.0}};

auto add = A + B; // element-wise addition
auto sub = A - B; // element-wise subtraction
auto mul = A * B; // Hadamard (element-wise) multiplication
auto dot = A % B; // matrix multiplication (dot product)
```

Beyond basic arithmetic, Usul also supports activation functions (`ReLU`, `Sigmoid`) that automatically register their derivatives into the compute graph.

## Getting Started

You will need a C++23 compiler, CMake (3.20+), and `vcpkg` to pull `xtensor` and `xtensor-blas`.

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset vcpkg-default
cmake --build build
./build/Usul
```
