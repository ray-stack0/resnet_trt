cmake_minimum_required(VERSION 3.5)

project(resnet50)

add_definitions(-std=c++11)

set(TARGET_NAME "resnet50")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
# message(STATUS ${PROJECT_SOURCE_DIR})
# include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.3/include/) # 头文件
link_directories(/usr/local/cuda-11.3/lib64/) # 库文件
# tensorrt
include_directories(/home/nvidia/环境配置/TensorRT-8.2.1.8/include/)
link_directories(/home/nvidia/环境配置/TensorRT-8.2.1.8/lib/)

# OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# 自己写的头文件
include_directories(${PROJECT_SOURCE_DIR}/include/*.h)

FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp )

add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} nvinfer)
target_link_libraries(${TARGET_NAME} cudart)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})
add_definitions(-O2 -pthread)

