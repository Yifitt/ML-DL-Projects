cmake_minimum_required(VERSION 3.31)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

# Eigen kütüphanesi
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(/mingw64/include/eigen3)

# Executable
add_executable(NeuralNetwork
        main.cpp
        layer.h
        functions.h
        neural_network.h
        utils.h

)

# Link libraries
target_link_libraries(NeuralNetwork Eigen3::Eigen)