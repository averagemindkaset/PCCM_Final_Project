#include "tri.h"

/***************************************************************************************************
void preProcessor::prepareMesh()
****************************************************************************************************
Does everythin' realted to mesh generation and transfers between processors.
***************************************************************************************************/
void triMesh::prepareMesh(inputSettings* settings)
{
    int mype;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    if (mype==0) cout << endl << "====================== MESH ======================" << endl;

    readMeshFiles(settings);
    formLocalNodeList();
    localizeNodeCoordinates();
    prepareBufferMatrices();

    return;
}

/***************************************************************************************************
void preProcessor::readMeshFiles()
****************************************************************************************************
File read procedure :
1- Name of the file to be opened is retrieved from the inputSetting obj.
2- File is opened in appropriate format, this is ascii format for minf or
   other text files and binary format for binary mesh files.
3- Read operation for minf file is straight forward. Binary files are read
   as size of a double or int and stored in readStream. Then swapbytes
   function is called to swap the bytes for the correct endianness.
4- Finally obtained data is deep-copied to the mesh data structure.
***************************************************************************************************/
void triMesh::readMeshFiles(inputSettings* settings)
{
    ifstream    file;            // file name object for serial file read
    string      dummy;           // dummy string to hold file names etc.
    char*       readStream;      // temperory var used for strings read from files
    double      dummyDouble;     // temperory var used for double values read from files

    /// these are general mpi parameters
    int mype, npes;              // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    /// these guys are used for parallel file input
    MPI_Status status;
    MPI_Offset offset;           // offset from the beginning of file for parallel file read
    MPI_Datatype mxyzftype,mienftype,mrngftype;        // mpi datatype used in parallel file read
    MPI_File fileptr;            // file pointer for parallel file read

    /***********************************************************************************************
    * READ THE MINF FILE
    * This file should hold the number of elements, nodes, space dimensions, element nodes and
    * element faces.
    ***********************************************************************************************/
    dummy = settings->getMinfFile();
    file.open(dummy.c_str(), ios::in);
    if (file.is_open()==false)
    {
        cout << "Unable to open minf file for pe: " << mype << "! Aborting... " << endl;
        MPI_Finalize();
        exit(0);
    }
    file >> dummy >> nn;
    file >> dummy >> ne;
    if (mype==0)
    {
        cout << "> Number of mesh elements : " << ne << endl;
        cout << "> Number of nodes : " << nn << endl;
        cout << "> File read complete: minf" << endl;
    }
    file.close();

    // Determine nec, mec
    nec = (ne-1)/npes + 1;
    mec = nec;
    if ((mype+1)*mec > ne)
        nec = ne - mype*mec;
    if (nec < 0)
        nec = 0;

    // Determine nnc, mnc
    nnc = (nn-1)/npes + 1;
    mnc = nnc;
    if ((mype+1)*mnc > nn)
        nnc = nn - mype*mnc;
    if (nnc < 0)
        nnc = 0;
    cout << "mype:" << mype << " nnc:" << nnc << " mnc:" << mnc << " nec:" << nec << endl;

    //Allocation of memory for the mesh data structure
    xyz = new double [nnc*nsd];
    massG = new double [nnc];
    MTnewG = new double [nnc];
    TG = new double [nnc];
    for(int i=0; i<nnc; i++){
        massG[i] = 0;
        MTnewG[i] = 0;
        TG[i] = 0;
    }

    node = new triNode[nnc];
    elem = new triElement[nec];
    ME      = new triMasterElement[nGQP];
    ME->setupGaussQuadrature();
    ME->evaluateShapeFunctions();
    if (mype==0) cout << "> Mesh data structure is created." << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    /***********************************************************************************************
    * READ THE MXYZ FILE
    * This file contains the node coordinates
    ***********************************************************************************************/
    dummy = settings->getMxyzFile();
    char * writable = new char[dummy.size() + 1];
    std::copy(dummy.begin(), dummy.end(), writable);
    writable[dummy.size()] = '\0';

    offset = mype*nsd*mnc*sizeof(double);
    MPI_Type_contiguous(nnc*nsd, MPI_DOUBLE, &mxyzftype);
    MPI_Type_commit(&mxyzftype);
    MPI_File_open(MPI_COMM_WORLD, writable, MPI_MODE_RDONLY, MPI_INFO_NULL, &fileptr);
    MPI_File_set_view(fileptr, offset, MPI_DOUBLE, mxyzftype, "native", MPI_INFO_NULL);
    readStream = new char [nsd*nnc*sizeof(double)];
    MPI_File_read(fileptr,readStream, nsd*nnc, MPI_DOUBLE, &status);
    swapBytes(readStream, nsd*nnc, sizeof(double));
    for(int i=0; i<nnc; i++)
    {
        node[i].setX(*((double*)readStream + nsd*i));
        node[i].setY(*((double*)readStream + nsd*i+1));
        xyz[i*nsd+xsd] = *((double*)readStream + nsd*i);
        xyz[i*nsd+ysd] = *((double*)readStream + nsd*i+1);
    }
    if (mype==0) cout << "> File read complete: " << dummy << endl;

    MPI_File_close(&fileptr);
    MPI_Barrier(MPI_COMM_WORLD);

    /***********************************************************************************************
    * READ THE MIEN FILE
    * This file contains the element connectivity
    ***********************************************************************************************/
    dummy = settings->getMienFile();
    writable = new char[dummy.size() + 1];
    std::copy(dummy.begin(), dummy.end(), writable);
    writable[dummy.size()] = '\0';

    offset = mype*nen*mec*sizeof(int);
    MPI_Type_contiguous(nec*nen, MPI_INT, &mienftype);
    MPI_Type_commit(&mienftype);
    MPI_File_open(MPI_COMM_WORLD, writable, MPI_MODE_RDONLY, MPI_INFO_NULL, &fileptr);
    MPI_File_set_view(fileptr, offset, MPI_INT, mienftype, "native", MPI_INFO_NULL);
    readStream = new char [nen*nec*sizeof(int)];
    MPI_File_read(fileptr, readStream, nen*nec, MPI_INT, &status);
    swapBytes(readStream, nen*nec, sizeof(int));
    int connValue;

    for(int i=0; i<nec; i++)
    {
        for(int j=0; j<nen; j++)
        {
            connValue = *((int*)readStream + nen*i+j);
            elem[i].setConn(j, connValue-1);
        }
    }

    if (mype==0) cout << "> File read complete: " << dummy << endl;

    MPI_File_close(&fileptr);
    MPI_Barrier(MPI_COMM_WORLD);

    dummyDouble = settings->getInitT();
    for(int i=0; i<nnc; i++){
        node[i].setT(dummyDouble);
        TG[i] = dummyDouble;
    }

    return;
}

void triMesh::swapBytes (char *array, int nelem, int elsize)
{
    register int sizet, sizem, i, j;
    char *bytea, *byteb;
    sizet = elsize;
    sizem = sizet - 1;
    bytea = new char [sizet];
    byteb = new char [sizet];
    for (i = 0; i < nelem; i++)
    {
        memcpy((void *)bytea, (void *)(array+i*sizet), sizet);
        for (j = 0; j < sizet; j++)
            byteb[j] = bytea[sizem - j];
        memcpy((void *)(array+i*sizet), (void *)byteb, sizet);
    }
    delete[] bytea;
    delete[] byteb;

    return;
}

/***************************************************************************************************
void triMesh::formLocalConnectivity()
****************************************************************************************************
Every processor have only a portion of the elements and a portion of the connectivity.
Nodal info stored in a process is not necessarly belong to the elements stored in that p.
Therefore, we need to collect the necessary node information from other processes.
In order to do this, a local node list is produced with this function.

    7____8____9                                3____4____5
    |\   |\   |        e0 = 0,1,7              |\   |\   |        e0 = 0,1,3
    | \  | \  |        e1 = 1,8,7        ==>   | \  | \  |        e1 = 1,4,3
    |  \ |  \ |        e2 = 1,2,8              |  \ |  \ |        e2 = 1,2,4
    |___\|___\|        e3 = 2,9,8              |___\|___\|        e3 = 2,5,4
    0     1      2                             0    1    2

    Step 1:
    allLocalNodes = 0,1,7,1,8,7,1,2,8,2,9, 8
    index         = 0,1,2,3,4,5,6,7,8,9,10,11

    Step 2: sorts
    allLocalNodes = 0,1,1,1,2,2,7,7,8,8,8, 9
    index         = 0,1,3,6,7,9,2,5,4,8,11,10

    Step 3:
    nnl = 6
    rawLocalConn  = 0,1,1,1,2,2,3,3,4,4,4,5

    Step 4:
    nodeLToG = 0,1,2,7,8,9

    Step 5:
    allLocalNodes = 0,1,3,1,4,3,1,2,4,2,5,4

***************************************************************************************************/
void triMesh::formLocalNodeList()
{
    int mype, npes;                // my processor rank and total number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    int n_allLocalNodes = nec*nen;                      //number of all local nodes
    int * allLocalNodes = new int [n_allLocalNodes];    //array for all local node values
    int * index = new int [n_allLocalNodes];
    int * rawLocalConn = new int [n_allLocalNodes];

    /*
     * 1. Prepare allLocalNodes
     * All the node numbers of all the elements on processor are gathered in a single array.
     * Obviously, there are some duplicate node values in this array. We will get rid of the
     * duplicates by first sorting this array and then selecting the unique values.
    */
    for(int i=0; i<nec; i++)
        for(int j=0; j<nen; j++)
            allLocalNodes[i*nen+j] = elem[i].getConn(j);

    /*
     * 2. Sort allLocalNodes
     * Now we can sort our allLocalNodes array.
     * An index array accompanying the allLocalNodes array will also be sorted.
     * With the index array it is easier to form the local connectivity later.
     */
    for(int i=0; i<n_allLocalNodes; i++)
        index[i] = i;
    quickSort(allLocalNodes, index, 0, n_allLocalNodes-1);

    /*
     * 3. Determine nnl (number of unique local nodes) and form raw connectivity numbering.
     * This means that for each node in the sorted allLocalNodes array, we give a local node number.
     */
    nnl = 1;
    rawLocalConn[0] = 0;
    for(int i=1; i<n_allLocalNodes; i++)
    {
        if (allLocalNodes[i-1] != allLocalNodes[i])
            nnl++;
        rawLocalConn[i] = nnl-1;
    }

    /*
     * 4. Collect local to global node number converter: nodeLToG (ieng)
     * Now we know the number of unique local nodes(nnl). We can collect the unique node values.
     */
    nodeLToG = new int [nnl];            //allocate space for nodeLToG array
    nodeLToG[0] = allLocalNodes[0];        //first value is already known
    int i_unique = 0;                    //iterator on nodeLToG
    for(int i=1; i<n_allLocalNodes; i++)
        if (nodeLToG[i_unique] != allLocalNodes[i])
        {
            i_unique++;
            nodeLToG[i_unique] = allLocalNodes[i];
        }

    /*
     * 5. Form local connectivity
     * Using the index array, allLocalNodes is overwritten with rawLocalConn
     */
    for(int i=0; i<n_allLocalNodes; i++)
        allLocalNodes[index[i]] = rawLocalConn[i];

    //here a new local level triNode structure is created.
    lNode = new triNode [nnl];

    // finally element level lConn array is filled with the allLocalNodes
    for(int i=0; i<nec; i++)
        for(int j=0; j<nen; j++)
            elem[i].setLConn(j,allLocalNodes[i*nen+j]);

/*    if (mype==0)
    for(int i=0; i<nec; i++)
    {
        for(int j=0; j<nen; j++)
            cout << elem[i].getLConn(j) << '\t';
        cout << endl;
    }
*/
    return;
}

/***************************************************************************************************
void triMesh::localizeNodeCoordinates()
****************************************************************************************************
We need to localize the necessary node connectivity using MPI_Get
partition node level -> local node level
***************************************************************************************************/
void triMesh::localizeNodeCoordinates()
{
    int n; //counter
    int in;     // node number in global
    int ipe;    // pe number where 'in' lies
    int inc;    // offset of in on ipe
    int npes;

    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    double * buffer;
    buffer = new double [mnc*nsd];
    int nncTarget;

    MPI_Win win;

    MPI_Win_create(xyz, nnc*nsd*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);

    for(int ipes=0; ipes<npes; ipes++)
    {
        for (int i = 0; i < mnc*nsd; i++) buffer[i] = 0;
        if ((ipes+1)*mnc > nn) nncTarget = nn - ipes*mnc;
        else nncTarget = mnc;
        // cout << "ipes" << ipes << "npes" << npes << endl;
        // cout << "mype " << mype << " ipes " << ipes << " npes " << npes << " nncTarget " << nncTarget << endl;
        MPI_Get(&buffer[0], nncTarget*nsd, MPI_DOUBLE, ipes, 0, nncTarget*nsd, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);

        for(int inl=0; inl<nnl; inl++)
        {
            in = nodeLToG[inl];
            ipe = in/mnc;
            inc = in%mnc;
            if (ipes == ipe)
            {
                lNode[inl].setX(buffer[inc*nsd+xsd]);
                lNode[inl].setY(buffer[inc*nsd+ysd]);
            } 
            // TL[inl] = buffer[inc];
        }
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    delete[] buffer;

    return;
}

// void triMesh::localizeNodeCoordinates()
// {
//     int n; //counter
//     int in;     // node number in global
//     int ipe;    // pe numnber where 'in' lies
//     int inc;    // offset of in on ipe

//     MPI_Win win;
//     double sendBufT;
//     double CoordGetx[nnl];
//     double CoordGety[nnl];

//     // all with MPI_Get
//     MPI_Win_create(xyz, nnc*nsd*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
//     MPI_Win_fence(0, win);
//     for(int i=0; i<nnl; i++)
//     {
//         in = nodeLToG[i];
//         ipe = in/mnc;
//         inc = in%mnc;
//         MPI_Get(&CoordGetx[i], 1, MPI_DOUBLE, ipe, inc*nsd+xsd, 1, MPI_DOUBLE, win);
//         MPI_Get(&CoordGety[i], 1, MPI_DOUBLE, ipe, inc*nsd+ysd, 1, MPI_DOUBLE, win);

//     }
//     MPI_Win_fence(0, win);
//     MPI_Win_free(&win);

//     for(int i=0; i<nnl; i++)
//     {
//         lNode[i].setX(CoordGetx[i]);
//         lNode[i].setY(CoordGety[i]);
//     }

//     return;
// }


/***************************************************************************************************
void triMesh::quickSort()
****************************************************************************************************
Necessary while forming of the local connectivity array.
***************************************************************************************************/
void triMesh::quickSort(int* arr, int* index, int left, int right)
{
    int i = left, j = right;
    int tmp, tmpIndex;
    int pivot = arr[(left + right) / 2];

    /* partition */
    while (i <= j)
    {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j)
        {
            tmp = arr[i];       tmpIndex = index[i];
            arr[i] = arr[j];    index[i] = index[j];
            arr[j] = tmp;       index[j] = tmpIndex;
            i++;
            j--;
        }
    };

    /* recursion */
    if (left < j)
        quickSort(arr, index, left, j);
    if (i < right)
        quickSort(arr, index, i, right);

    return;
}

/***************************************************************************************************
void triMesh::prepareBufferMatrices()
****************************************************************************************************
Instead of creating and destroying buffer matrices at each iteration, it is best to keep them at
hand in the triMesh structure. Can be further simplified.
***************************************************************************************************/
void triMesh::prepareBufferMatrices()
{
    //Allocation of memory for the mesh data structure
    MTnewL = new double [nnl];
    TL = new double [nnl];
    for (int i=0; i<nnl;i++){
        MTnewL[i] = 0;
        TL[i] =0;
    }

    return;
}

/***************************************************************************************************
GAUSS QUADRATURE PIONTS AND WEIGHTS ARE SET FOR 7 POINT QUADRATUE FORMULA
***************************************************************************************************/

void triMasterElement::setupGaussQuadrature()
{
    this[0].point[0] = 0.333333333333333;
    this[0].point[1] = 0.333333333333333;
    this[0].weight   = 0.225 / 2.0;

    this[1].point[0] = 0.059715871789770;
    this[1].point[1] = 0.470142064105115;
    this[1].weight   = 0.132394152788 / 2.0;

    this[2].point[0] = 0.470142064105115;
    this[2].point[1] = 0.059715871789770;
    this[2].weight   = 0.132394152788 / 2.0;

    this[3].point[0] = 0.470142064105115;
    this[3].point[1] = 0.470142064105115;
    this[3].weight   = 0.132394152788 / 2.0;

    this[4].point[0] = 0.101286507323456;
    this[4].point[1] = 0.797426985353087;
    this[4].weight   = 0.125939180544 / 2.0;

    this[5].point[0] = 0.101286507323456;
    this[5].point[1] = 0.101286507323456;
    this[5].weight   = 0.125939180544 / 2.0;

    this[6].point[0] = 0.797426985353087;
    this[6].point[1] = 0.101286507323456;
    this[6].weight   = 0.125939180544 / 2.0;

    return;
}

/***************************************************************************************************
EVALUATES SHAPE FUNCTIONS FOR LINEAR TRIANGULAR ELEMENT
***************************************************************************************************/
void triMasterElement::evaluateShapeFunctions()
{
    double ksi;
    double eta;

    for(int i=0; i<nGQP; i++)
    {
        ksi  = this[i].point[0];
        eta  = this[i].point[1];

        this[i].S[0] = 1.0-ksi-eta;
        this[i].S[1] = ksi;
        this[i].S[2] = eta;

        this[i].dSdKsi[0] = -1.0;
        this[i].dSdKsi[1] =  1.0;
        this[i].dSdKsi[2] =  0.0;

        this[i].dSdEta[0] = -1.0;
        this[i].dSdEta[1] =  0.0;
        this[i].dSdEta[2] =  1.0;
    }

    return;
}
