#ifndef SOLVER_H_
#define SOLVER_H_

#include "tri.h"


/***************************************************************************************************
 * Solver class
 ***************************************************************************************************/
class femSolver
{
private:
    // VARIABLES
    int nnSolved;               // Number of nodes solved (non-Dirichlet nodes)
    inputSettings*  settings;   // a local pointer for the settings
    triMesh*        mesh;       // a local pointer for the mesh

    // METHODS
    void calculateJacobian(const int);
    void calculateElementMatrices(const int);
    void applyDrichletBC();
    void accumulateMass();
    void explicitSolver();
    double calculateRealPosition(const double , const double , const double*);
    double calculateHeatSource(const double, const double);
protected:

public:
    // DEFAULT CONSTRUCTOR
    femSolver(){};
    // DESTRUCTOR
    ~femSolver(){};
    // INTERFACE METHOD
    void solverControl(inputSettings*, triMesh*);

};

#endif /* SOLVER_H_ */
