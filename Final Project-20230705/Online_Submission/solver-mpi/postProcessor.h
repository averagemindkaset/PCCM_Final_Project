#ifndef POSTPROCESSOR_H_
#define POSTPROCESSOR_H_

#include "solver.h"

#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkSmartPointer.h>

class postProcessor
{
    private:
        //VARIABLES
        inputSettings*  settings;   // a local pointer for the settings
        triMesh*        mesh;       // a local pointer for the mesh
        double          partialminT;       // min value of the Temperature field
        double          partialmaxT;

        //METHODS
        void evaluateLimits();
        void vtkVisualization();
        void compareAnalytical();
    protected:

    public:
        void postProcessorControl(inputSettings*, triMesh*);
};


#endif /* POSTPROCESSOR_H_ */
