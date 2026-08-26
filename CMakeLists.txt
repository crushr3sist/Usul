cmake_minimum_required(VERSION 3.20)

project(Usul VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH vcpkg_installed/x64-windows/)

# 1. Find the package
find_package(xtensor REQUIRED)

# 2. Create the target FIRST
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

# 3. Then link and add includes to the target
target_include_directories(Usul PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(Usul PUBLIC xtensor)

target_include_directories(
    Usul PRIVATE
    include
)