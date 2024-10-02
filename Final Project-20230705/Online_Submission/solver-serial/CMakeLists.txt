cmake_minimum_required(VERSION 3.2)
PROJECT(2d_Unsteady)

set(VTK_DIR "/usr/lib64/vtk")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
include_directories(.)
file(GLOB SOURCES "*.cpp")

add_executable(2d_Unsteady ${SOURCES})

target_link_libraries(2d_Unsteady ${VTK_LIBRARIES})
