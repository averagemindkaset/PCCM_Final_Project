# 2D Unsteady Heat Equation Solver

To compile the code perform the following steps from within `skeleton` directory.

``` bash
    mkdir build && cd build             # Create build folder and navigate into it.
    cmake .. -DBUILD_SELECTOR=<build>   # Run CMake for the specified build.
    make                                # Perform the compilation of the solver.
```

In the commands above, make sure to replace `<build>` with one of the four build
options listed below:

``` bash
    cmake .. -DBUILD_SELECTOR=Serial        # Compile serial solver.
    cmake .. -DBUILD_SELECTOR=OpenMP_Task_A # Compile parallel OpenMP solver for task A.
    cmake .. -DBUILD_SELECTOR=OpenMP_Task_B # Compile parallel OpenMP solver for task B.
    cmake .. -DBUILD_SELECTOR=MPI           # Compile parallel MPI solver.
```

After performing the `make` command, executable named `2d_Unsteady_<build>` is
generated within the `build` folder.
