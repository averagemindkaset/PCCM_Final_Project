cmake_minimum_required(VERSION 2.8)
PROJECT(2d_Unsteady)

# Require MPI for this project:
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
# add_compile_options(-g -ftrapv)
add_compile_options(-O3)

set(VTK_DIR "/usr/lib64/vtk")
# set(VTK_DIR "/usr/include/vtk-7.1")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
include_directories(.)
file(GLOB SOURCES "*.cpp")

add_executable(2d_Unsteady ${SOURCES})

target_link_libraries(2d_Unsteady ${VTK_LIBRARIES})
target_link_libraries(2d_Unsteady ${MPI_C_LIBRARIES})

