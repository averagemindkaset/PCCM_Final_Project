/***************************************************************************************************
Name        : 2D_Unsteady_Diffusion.cpp
***************************************************************************************************/

#include "settings.h"
#include "tri.h"
#include "solver.h"
#include "postProcessor.h"

int main(int argc, char **argv) {
/***************************************************************************************************
MAIN PROGRAM FLOW
1. Pre-Processing Stage
    1.1. Settings
    1.2. Mesh
2. Solution Stage
3. Post-Processing Stage
***************************************************************************************************/

    int mype, npes;
    double starttime;

    inputSettings*  settings    = new inputSettings(argc, argv);
    triMesh*        mesh        = new triMesh;
    femSolver*      solver      = new femSolver;
    postProcessor*  postP       = new postProcessor;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    if(mype == 0) cout << "A Parallel 2-Dimensional Unsteady Diffusion Equation Solver"  << endl;

    // Pre-Processing Stage
    starttime = MPI_Wtime(); //timer starts
    settings->prepareSettings();
    mesh->prepareMesh(settings);

    // Solution Stage
    solver->solverControl(settings, mesh);

    // Post-Processing Stage
    postP->postProcessorControl(settings, mesh);
    if(mype==0) cout << "Elapsed time is " << fixed << MPI_Wtime()-starttime << endl;

    // Cleanup
    delete settings;
    delete mesh;
    delete solver;
    delete postP;
    if(mype == 0) cout << ": Ciao:)"  << endl;
    MPI_Finalize();

    return 0;
}
