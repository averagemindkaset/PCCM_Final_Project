#ifndef TRI_H_
#define TRI_H_

#include "settings.h"

/***************************************************************************************************
7 POINT GAUSS QUADRATURE INTEGRATION - LINEAR TRIANGULAR MASTER ELEMENT
***************************************************************************************************/
class triMasterElement
{
    private:
        // VARIABLES
        double point[2];    // ksi and eta for each GQ point
        double weight;      // weight of each GQ point
        double S[3];        // Shape functions
        double dSdKsi[3];   // ksi derivatives of shape functions
        double dSdEta[3];   // eta derivatives of shape functions
    protected:

    public:
        //GETTERS
        double getPoint(int i)    {return point[i];};
        double getWeight()        {return weight;};
        double getS(int i)        {return S[i];};
        double getDSdKsi(int i)   {return dSdKsi[i];};
        double getDSdEta(int i)   {return dSdEta[i];};
        //INTERFACE
        void setupGaussQuadrature();
        void evaluateShapeFunctions();
};

/***************************************************************************************************
NODE LEVEL DATA STRUCTURE
****************************************************************************************************
Each node has
    - coordinates
    - a variable
Comments:
    - For the current application, the only variable is temperature since we are solving transient
      diffusion equation.
***************************************************************************************************/
class triNode
{
    private:
        friend class femSolver;
        // VARIABLES
        double x;    // x-coordinate
        double y;    // y-coordinate
        double T;    // Temperature
        double MTnew;
        double mass;
        int BCtype;
    protected:

    public:
        // DEFAULT CONSTRUCTOR
        triNode(){x=0.0f; y=0.0f; T=0.0f; MTnew=0.0f; mass=0.0f; BCtype=0;};
        // SETTERS
        inline void setX         (double value)    {x = value;};
        inline void setY         (double value)    {y = value;};
        inline void setT         (double value)    {T = value;};
        inline void setMTnew     (double value)    {MTnew = value;};
        inline void addMTnew     (double value)    {MTnew = MTnew + value;};
        inline void setMass      (double value)    {mass = value;};
        inline void addMass      (double value)    {mass = mass + value;};
        inline void setBCtype    (int value)       {BCtype = value;}
        // GETTERS
        inline double    getX()       {return x;};
        inline double    getY()       {return y;};
        inline double    getT()       {return T;};
        inline double    getMTnew()   {return MTnew;};
        inline double    getMass()    {return mass;};
        inline int       getBCtype()  {return BCtype;};
};

/***************************************************************************************************
ELEMENT LEVEL DATA STRUCTURE
****************************************************************************************************
An element has
    - connectivity [nen]
    - boundary condition [nef]
    - Jacobian determinant [nGQP]
    - dsdX [3xnGQP]
    - dSdY [3xnGQP]
***************************************************************************************************/
class triElement
{
    private:
        // VARIABLES
        int conn[nen];
        int lConn[nen];
        struct{
            double detJ;
            double dSdX[3];
            double dSdY[3];
        }GQP[nGQP];
        double M[nen];
        double F[nen];
        double K[nen][nen];
    protected:

    public:
        // DEFAULT CONSTRUCTOR
        // I must aboulutly write a proper constructor here :)
        // SETTERS
        void setConn    (int i, int value)              {conn[i] = value;};
        void setLConn   (int i, int value)              {lConn[i] = value;};
        void setDetJ    (int i, double value)           {GQP[i].detJ = value;};
        void setDSdX    (int i, int j, double value)    {GQP[i].dSdX[j] = value;};
        void setDSdY    (int i, int j, double value)    {GQP[i].dSdY[j] = value;};
        void setK       (int i, int j, double value)    {K[i][j]=value;};
        void setM       (int i, double value)           {M[i]=value;};
        void setF       (int i, double value)           {F[i]=value;};
        void addF       (int i, double value)           {F[i]=F[i]+value;}; 
        // GETTERS
        int           getConn (int index)       {return conn[index];};
        int           getLConn(int index)       {return lConn[index];};
        double        getDetJ (int i)           {return GQP[i].detJ;};
        double        getDSdX (int i, int j)    {return GQP[i].dSdX[j];};
        double        getDSdY (int i, int j)    {return GQP[i].dSdY[j];};
        double        getK    (int i, int j)    {return K[i][j];};
        double        getM    (int i)           {return M[i];};
        double        getF    (int i)           {return F[i];};
        double *      getKptr ()                {return &K[0][0];};
        double *      getMptr ()                {return M;};
        double *      getFptr ()                {return F;};
};

/***************************************************************************************************
MESH DATA STRUCTURE
****************************************************************************************************
This class is used to keep the pointers to node and element data structures. It is more
convinient to pass the pointer to this class rather than passin the pointers to both element and
node arrays during function calls. So it is just for simplification.
***************************************************************************************************/
class triMesh
{
    private:
        // VARIABLES
        int ne;                     // total number of elements
        int nec;                    // number of elements per cpu
        int mec;                    // max. number of elements among all cpus
        int nn;                     // number of nodes
        int nnc;                    // number of nodes per core
        int nnl;                    // number of local nodes
        int mnc;                    // max. number of nodes among all cpus
        int * nodeLToG;             // array for local to global connectivity conversion
        triNode*            node;   // pointer to partition, node level data structure
        triNode*            lNode;  // pointer to local, node level data structure
        triElement*         elem;   // pointer to element level data structure
        triMasterElement*   ME;     // pointer to master element data structure

        double * xyz;
        double * massG;
        double * MTnewG;
        double * MTnewL;
        double * TG; // size nnc
        double * TL; // size nnl


        //METHODS
        void readMeshFiles(inputSettings*);
        void swapBytes(char*, int, int);    // Used during file read operations
        void formLocalNodeList();
        void quickSort(int*, int*, int, int);
        void localizeNodeCoordinates();
        void prepareBufferMatrices();
    protected:

    public:
        // DEFAULT CONSTRUCTOR
        triMesh(){};
        // DESTRUCTOR
        ~triMesh()
        {
            delete[] node;    // Recovers memory for node level data structure.
            delete[] lNode;   // Recovers memory for local node level data structure.
            delete[] elem;    // Recovers memory for element level data structure.
            delete[] ME;      // Recovers memory for master element data structure.
        };
        // GETTERS
        int        getNe()                 {return ne;};
        int        getNec()                {return nec;};
        int        getMec()                {return mec;};
        int        getNn()                 {return nn;};
        int        getNnc()                {return nnc;};
        int        getNnl()                {return nnl;};
        int        getMnc()                {return mnc;};
        int*       getNodeLToG()           {return nodeLToG;};

        double*    getXyz()                {return xyz;};
        double*    getMassG()              {return massG;};
        double*    getMTnewG()             {return MTnewG;};
        double*    getMTnewL()             {return MTnewL;};
        double*    getTG()                 {return TG;};
        double*    getTL()                  {return TL;};

        triNode*           getNode     (int index)    {return &node[index];};
        triNode*           getLNode    (int index)    {return &lNode[index];};
        triElement*        getElem     (int index)    {return &elem[index];};
        triMasterElement*  getME       (int index)    {return &ME[index];};

        // INTERFACE METHOD
        void prepareMesh(inputSettings*);
};

#endif /* TRI_H_ */
