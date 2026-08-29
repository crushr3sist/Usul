cmake_minimum_required(VERSION 3.20)

project(Usul VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenBLAS CONFIG REQUIRED)
# Find xtensor packages
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)

# Create executable
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
target_link_libraries(Usul PRIVATE OpenBLAS::OpenBLAS)

# Link dependencies
target_link_libraries(
    Usul
    PRIVATE
        xtensor
        xtensor-blas
)

# Project headers
target_include_directories(
    Usul
    PRIVATE
        include
)