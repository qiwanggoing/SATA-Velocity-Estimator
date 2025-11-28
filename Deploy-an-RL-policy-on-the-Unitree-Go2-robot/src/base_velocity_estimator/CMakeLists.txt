cmake_minimum_required(VERSION 3.5)
project(base_velocity_estimator)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
# 1. 查找 Unitree 包
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
# 添加节点
# add_executable(imu_velocity_estimator_node src/imu_velocity_estimator_node.cpp)
# ament_target_dependencies(imu_velocity_estimator_node
#   rclcpp
#   sensor_msgs
#   geometry_msgs
# )
include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}  # 第三方库（如Eigen）
)
add_executable(ekf_velocity_estimator_node
src/main.cpp
src/ekf_velocity_estimator.cpp
src/kinematics_velocity_estimator_node.cpp)
# add_executable(kinematics_velocity_estimator_node src/kinematics_velocity_estimator_node.cpp)
ament_target_dependencies(ekf_velocity_estimator_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  unitree_go
  unitree_api
)
target_include_directories(ekf_velocity_estimator_node PRIVATE
  ${pinocchio_INCLUDE_DIRS}
)
target_link_libraries(ekf_velocity_estimator_node
  Eigen3::Eigen  # 显式链接 Eigen
  ${pinocchio_LIBRARIES}
)

# 安装节点
# install(TARGETS imu_velocity_estimator_node
#   DESTINATION lib/${PROJECT_NAME}
# )
# install(TARGETS kinematics_velocity_estimator_node
#   DESTINATION lib/${PROJECT_NAME}
# )
install(TARGETS ekf_velocity_estimator_node
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS
    scripts/mujoco_simulator.py
    scripts/xbox_command.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
