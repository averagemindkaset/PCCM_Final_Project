cmake_minimum_required(VERSION 3.2)
PROJECT(2d_Unsteady)

# Add compiler options in this clause
add_compile_options(-O0)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # require at least gcc 4.8
    message("TESTING")
    
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.1)
        message(
            "\n***********************************************\n"
            "***********************************************\n"
            "***                                         ***\n"
            "***  Wrong compiler selected!               ***\n"
            "***                                         ***\n"
            "***  Please use Intel 19.1 for compiling    ***\n"
            "***  -----------------------------------    ***\n"
            "***                                         ***\n"
            "***  To check the current compiler:         ***\n"
            "***  `module list`                          ***\n"
            "***                                         ***\n"
            "***  To unload the current compiler:        ***\n"
            "***  `module unload <compiler>`             ***\n"
            "***                                         ***\n"
            "***  To load the supported compiler:        ***\n"
            "***  `module load intel/19.1`               ***\n"
            "***                                         ***\n"
            "***********************************************\n"
            "***********************************************\n")
            message(FATAL_ERROR "Fatal Error!")
    endif()
else()
        message(
            "\n***********************************************\n"
            "***********************************************\n"
            "***                                         ***\n"
            "***  Wrong compiler selected!               ***\n"
            "***                                         ***\n"
            "***  Please use Intel 19.1 for compiling    ***\n"
            "***  -----------------------------------    ***\n"
            "***                                         ***\n"
            "***  To check the current compiler:         ***\n"
            "***  `module list`                          ***\n"
            "***                                         ***\n"
            "***  To unload the current compiler:        ***\n"
            "***  `module unload <compiler>`             ***\n"
            "***                                         ***\n"
            "***  To load the supported compiler:        ***\n"
            "***  `module load intel/19.1`               ***\n"
            "***                                         ***\n"
            "***********************************************\n"
            "***********************************************\n")
            message(FATAL_ERROR "Fatal Error!")
endif()





# Compile build options Serial/MPI/OpenMP A,B
set(BUILD_SELECTOR "Option A" CACHE STRING "Help message for this variable")
set_property(
  CACHE BUILD_SELECTOR
  PROPERTY STRINGS
  "Serial" "OpenMP_Task_A" "OpenMP_Task_B" "MPI"
)

# Include VTK library
set(VTK_DIR "/usr/lib64/vtk")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
include_directories(.)


# Build switchboard

# Build Serial Case
if(BUILD_SELECTOR STREQUAL "Serial" )
    
message(
    "\n***********************************************\n"
    "***********************************************\n"
    "***                                         ***\n"
    "***  Compiled for execution in serial       ***\n"
    "***                                         ***\n"
    "***********************************************\n"
    "***********************************************\n")

    file(GLOB SOURCES "./solver-serial/*.cpp")
    
    add_executable(2d_Unsteady_Serial ${SOURCES})
    target_link_libraries(2d_Unsteady_Serial ${VTK_LIBRARIES})

# Build OpenMP Critical Case
elseif (BUILD_SELECTOR STREQUAL "OpenMP_Task_A" )

    message(
        "\n***********************************************\n"
        "***********************************************\n"
        "***                                         ***\n"
        "***  Compiled for execution using OpenMP    ***\n"
        "***  in Task A.                             ***\n"
        "***                                         ***\n"
        "***********************************************\n"
        "***********************************************\n")

    # Require OpenMP for this project:
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    file(GLOB SOURCES "./solver-openmp-a/*.cpp")
    
    add_executable(2d_Unsteady_OpenMP_A ${SOURCES})
    target_link_libraries(2d_Unsteady_OpenMP_A ${VTK_LIBRARIES})

# Build OpenMP Reduction Case
elseif (BUILD_SELECTOR STREQUAL "OpenMP_Task_B" )
    
    message(
        "\n***********************************************\n"
        "***********************************************\n"
        "***                                         ***\n"
        "***  Compiled for execution using OpenMP    ***\n"
        "***  in Task B.                             ***\n"
        "***                                         ***\n"
        "***********************************************\n"
        "***********************************************\n")

    # Require OpenMP for this project:
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    file(GLOB SOURCES "./solver-openmp-b/*.cpp")
    
    add_executable(2d_Unsteady_OpenMP_B ${SOURCES})
    target_link_libraries(2d_Unsteady_OpenMP_B ${VTK_LIBRARIES})

# Build MPI Case
elseif (BUILD_SELECTOR STREQUAL "MPI" )
    
    message(
        "\n***********************************************\n"
        "***********************************************\n"
        "***                                         ***\n"
        "***  Compiled for execution using MPI.      ***\n"
        "***                                         ***\n"
        "***********************************************\n"
        "***********************************************\n")

    # Require MPI for this project:
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})

    file(GLOB SOURCES "./solver-mpi/*.cpp")

    add_executable(2d_Unsteady_MPI ${SOURCES})
    target_link_libraries(2d_Unsteady_MPI ${VTK_LIBRARIES})
    target_link_libraries(2d_Unsteady_MPI ${MPI_C_LIBRARIES})

# Help Message for Erroneous Input
else()

    message(
        "\n***********************************************\n"
        "***********************************************\n"
        "***                                         ***\n"
        "***  Specify build type:                    ***\n"
        "***  -DBUILD_SELECTOR=<build>               ***\n"
        "***                                         ***\n"
        "***  The following builds are available:    ***\n"
        "***  -----------------------------------    ***\n"
        "***  1. Serial                              ***\n"
        "***  2. OpenMP_Task_A                       ***\n"
        "***  3. OpenMP_Task_B                       ***\n"
        "***  4. MPI                                 ***\n"
        "***                                         ***\n"
        "***********************************************\n"
        "***********************************************\n")

    message(FATAL_ERROR "Could not compile see information above")

endif()

message(
    "\n***********************************************\n"
    "***********************************************\n"
    "***                                         ***\n"
    "***  CMake has completed...                 ***\n"
    "***  Run `make` to build the executable     ***\n"
    "***                                         ***\n"
    "***********************************************\n"
    "***********************************************\n")
