cmake_minimum_required(VERSION 3.14)
project(neural_net)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(neural_net main.cpp src/DenseLayer.cpp src/DenseLayer.h src/ActivationFunction.h src/ReLU.cpp src/ReLU.h src/DataReader.cpp src/DataReader.h src/Model.cpp src/Model.h src/LinearActivation.cpp src/LinearActivation.h src/Sigmoid.cpp src/Sigmoid.h)
target_link_libraries(neural_net ${ARMADILLO_LIBRARIES})