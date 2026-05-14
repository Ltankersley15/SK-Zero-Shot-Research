cmake_minimum_required(VERSION 3.8)
project(fr3_skill_server)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanExecuteJointGoal.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
