cmake_minimum_required(VERSION 3.2)
PROJECT(2d_Unsteady_OpenMP)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(VTK_DIR "/usr/lib64/vtk")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
include_directories(.)
file(GLOB SOURCES "*.cpp")

add_executable(2d_Unsteady_OpenMP ${SOURCES})

target_link_libraries(2d_Unsteady_OpenMP ${VTK_LIBRARIES})
