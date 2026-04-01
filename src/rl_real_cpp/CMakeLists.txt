cmake_minimum_required(VERSION 3.8)
project(rl_real_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# Find Torch
# 尝试多个可能的TorchConfig.cmake位置
set(TORCH_PATHS
  /home/woan/libtorch/share/cmake/Torch
  /home/woan/libtorch/share/cmake
  /home/woan/libtorch
)

foreach(PATH IN LISTS TORCH_PATHS)
  if(EXISTS "${PATH}/TorchConfig.cmake")
    set(Torch_DIR "${PATH}")
    message(STATUS "Found TorchConfig.cmake at: ${PATH}")
    break()
  endif()
endforeach()

if(NOT DEFINED Torch_DIR)
  message(FATAL_ERROR "TorchConfig.cmake not found in any of the following paths:\n${TORCH_PATHS}")
endif()

find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
  message(FATAL_ERROR "Torch not found. Please check your Torch installation.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch include directory: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# Find Eigen
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)

# Add executable
add_executable(rl_real src/rl_real.cpp)

# Link libraries
target_link_libraries(rl_real
  yaml-cpp
  ${TORCH_LIBRARIES}
)

# Handle ROS2 dependencies
ament_target_dependencies(rl_real
  rclcpp
  std_msgs
  sensor_msgs
)

# Install executable
install(TARGETS rl_real
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()