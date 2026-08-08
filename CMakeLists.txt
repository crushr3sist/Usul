cmake_minimum_required(VERSION 3.20)

project(Usul VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(
    Usul
    src/main.cpp
    src/neural/node.cpp
    src/neural/layer.cpp
    src/neural/neuron.cpp
    src/util/rand.cpp
    src/optim/autodiff.cpp
    src/optim/gradient_descent.cpp
)

target_include_directories(
    Usul PRIVATE
    include
)