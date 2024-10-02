#include "postProcessor.h"

/***************************************************************************************************
preProcessorControl
**************************************************************************************************/

void postProcessor::postProcessorControl(inputSettings* argSettings, triMesh* argMesh)
{
    int mype, npes;             // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    mesh = argMesh;
    settings = argSettings;

    if (mype==0) cout << endl << "================ POST-PROCESSING =================" << endl;

    evaluateLimits();
    compareAnalytical();
    vtkVisualization();

    return;
}

/***************************************************************************************************
Evaluates the maximum and minimum temperatures in the field
***************************************************************************************************/
void postProcessor::evaluateLimits()
{
    int nn = mesh->getNn();
    int nnc = mesh->getNnc();
    int mype, npes;             // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    double * TG = mesh->getTG();
    double T;
    double minT, maxT;
    partialminT = std::numeric_limits<double>::max();
    partialmaxT = std::numeric_limits<double>::min();

    for(int i=0; i<nnc; i++)
    {
        T = TG[i];
        if(T < partialminT)
            partialminT = T;
        if(T > partialmaxT)
            partialmaxT = T;
    }
    MPI_Reduce(&partialminT, &minT, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&partialmaxT, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mype == 0) cout << "Tmin " << minT << endl << "Tmax " << maxT << endl;

    return;
}

/***************************************************************************************************
Compare with analytical solution
***************************************************************************************************/
void postProcessor::compareAnalytical()
{
    int mype;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    int nnc = mesh->getNnc();
    double * TG = mesh->getTG();
    double * xyz = mesh->getXyz();
    double x, y, radius, Ta;
    triNode* pNode;     // temporary pointer to hold partition nodes

    // Outer temperature
    double T0 = 500;
    // Heat flux
    double Q = 100;
    // Thermal conductivity
    double k = 1;
    // Radius of the disk with source
    double Rs = 0.01;
    // Disk radius
    double Rd = 0.1;
    const double PI = std::atan(1.0)*4;
    double partialMSE = 0;
    double MSE = 0;
    int partialnnIn = 0;
    int nnIn = 0;

    for(int i=0; i<nnc; i++)
    {
        pNode = mesh->getNode(i);
        if(pNode->getBCtype() != 1)
        {
            x = xyz[i*nsd+xsd];
            y = xyz[i*nsd+ysd];
            radius = sqrt(pow(x,2) + pow(y,2));
            // cout << "radius " << radius << endl;
            if(radius < (Rs - 1e-10))
            {
                Ta = T0 - (Q / (2 * PI * k)) * (0.5 *(pow(radius,2) / pow(Rs,2) - 1) + log(Rs / Rd));
                // cout << "in " << log(Rs / Rd) << endl;
            }
            else
            {
                Ta = T0 - (Q / (2 * PI * k)) * log(radius / Rd);
                // cout << "out " << Ta << endl;
            }
            partialMSE += pow(TG[i] - Ta,2);
            partialnnIn += 1;
            // cout << "Sim result " << T[i] << " Th result " << Ta << endl;
            // Generate theoretical output
            // T[i] = Ta;
        }
    }

    MPI_Reduce(&partialMSE, &MSE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&partialnnIn, &nnIn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // MPI_Allreduce(&partialMSE, &MSE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MSE = sqrt(MSE/nnIn);
    if (mype == 0) cout << "RMS error " << MSE << endl;

    return;
}

/***************************************************************************************************
// Main visualization function
***************************************************************************************************/
void postProcessor::vtkVisualization()
{
    int nn = mesh->getNn();
    int nnc = mesh->getNnc();
    int nnl = mesh->getNnl();
    int mnc = mesh->getMnc();
    int ne = mesh->getNe();
    int nec = mesh->getNec();
    int mec = mesh->getMec();
    double * TL = mesh->getTL();

     int mype, npes;             // my processor rank and total number of processors

     MPI_Comm_rank(MPI_COMM_WORLD, &mype);
     MPI_Comm_size(MPI_COMM_WORLD, &npes);
     ostringstream int2str;
     int2str << mype;
     string strMype = int2str.str();
     string dummy;

     // VTK Double Array
     vtkSmartPointer<vtkDoubleArray> pcoords = vtkSmartPointer<vtkDoubleArray>::New();
     pcoords->SetNumberOfComponents(nen);
     pcoords->SetNumberOfTuples(nnl);

     //vtkDoubleArray type pcoords is filled with the data in meshPoints.
     for (int i=0; i<nnl; i++)
         pcoords->SetTuple3(i,mesh->getLNode(i)->getX(),mesh->getLNode(i)->getY(),0.0f);

     //vtkPoints type outputPoints is filled with the data in pcoords.
     vtkSmartPointer<vtkPoints> outputPoints = vtkSmartPointer<vtkPoints>::New();
     outputPoints->SetData(pcoords);

     //Connectivity is written to vtkCellArray type outputCells
     vtkSmartPointer<vtkCellArray> connectivity = vtkSmartPointer<vtkCellArray>::New();
     for(int i=0; i<nec; i++)
     {
         connectivity->InsertNextCell(nen);
         for(int j=0; j<nen; j++)
             connectivity->InsertCellPoint(mesh->getElem(i)->getLConn(j));
     }

     // Scalar property
     vtkSmartPointer<vtkDoubleArray> scalar = vtkSmartPointer<vtkDoubleArray>::New();
     scalar->SetName("Temperature");
     for(int i=0; i<nnl; i++)
         scalar->InsertNextValue(TL[i]);

     // Previously collected data which are outputPoints, outputCells, scalarProperty, are written to
     // vtkUnstructuredGrid type grid.
     vtkSmartPointer<vtkUnstructuredGrid> unsGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
     unsGrid->SetPoints(outputPoints);
     unsGrid->SetCells(5,connectivity);
     unsGrid->GetPointData()->SetScalars(scalar);

     //Whatever collected in unstructured grid above is written to the "Title_mype.vtu" file below.
     vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
     dummy = settings->getTitle();
     dummy.append("_");
     dummy.append(strMype);
     dummy.append(".vtu");
     cout << dummy << endl;
     writer->SetFileName(dummy.c_str());
     writer->SetInputData(unsGrid);
     writer->Write();

     // Now we write the "Title.pvtu" file which contains the informtaiton about other files.
     if(mype == 0)
     {
         vtkSmartPointer<vtkXMLPUnstructuredGridWriter> pwriter =
             vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
         dummy = settings->getTitle();
         dummy.append(".pvtu");
         pwriter->SetFileName(dummy.c_str());
         pwriter->SetNumberOfPieces(npes);
         #if VTK_MAJOR_VERSION <= 5
             pwriter->SetInput(unsGrid);
         #else
             pwriter->SetInputData(unsGrid);
         #endif
         pwriter->Write();
     }

     return;
 }
