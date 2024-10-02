#ifndef SETTINGS_H_
#define SETTINGS_H_

#include "constants.h"

/***************************************************************************************************
Boundary Condition specification class
***************************************************************************************************/
class bndc
{
    private:
        friend class inputSettings;
        int BCType;
        double BCValue1;
        double BCValue2;
    protected:

    public:
        //DEFAULT CONSTRUCTOR
        bndc(){BCType=0; BCValue1=0.0; BCValue2 = 0.0;};
        //SETTERS
        void setType    (int value)       {BCType = value;};
        void setValue1  (double value)    {BCValue1 = value;};
        void setValue2  (double value)    {BCValue2 = value;};
        //GETTERS
        int     getType()    {return BCType;};
        double  getValue1()  {return BCValue1;};
        double  getValue2()  {return BCValue2;};
};

/***************************************************************************************************
Input parameters are stored in this class. readSettingsFile() method does the job: It opens the
settings.in file, reads the parameters and assigns them to the private variables.
Why triMEsh is a friend? : triMesh method readMeshFiles needs to set ne and nn. no other funtions
need to change a private variable of inputSettings. Thus, previliges are given to triMesh.
***************************************************************************************************/

class inputSettings
{
    private:
        // VARIABLES
        int        argc;       // command line argument
        char**    argv;        // command line argument
        string    title;       // title of the document
        string    minfFile;    // information file name
        string    mxyzFile;    // node coordinates file name
        string    mienFile;    // connctivity file name
        string    mrngFile;    // boundary info file name
        string    dataFile;    // data file name
        double    initT;       // initial value of the temperature
        double    D;           // Diffusion coefficient
        double    source;      // Heat source term
        int       nFG;         // Number of face groups
        int       nIter;       // number of maximum time steps
        double    dt;          // time step size
        bndc*     BC;          // boundary conditions
        // METHODS
        void readSettingsFile();
        void printSettings();

    protected:

    public:
        /// CONSTRUCTORS ///
        inputSettings();
        inputSettings(int, char**);
        /// GETTERS ///
        int                getArgc()       {return argc;};
        char**            getArgv()        {return argv;};
        string            getTitle()       {return title;};
        string            getMinfFile()    {return minfFile;};
        string            getMxyzFile()    {return mxyzFile;};
        string            getMienFile()    {return mienFile;};
        string            getMrngFile()    {return mrngFile;};
        string            getDataFile()    {return dataFile;};
        double            getInitT()       {return initT;};
        double            getD()           {return D;};
        double            getSource()      {return source;};
        int               getNFG()         {return nFG;};
        bndc*             getBC(int i)     {return &BC[i];};
        int               getNIter()       {return nIter;};
        double            getDt()          {return dt;};

        //INTERFACE METHOD
        void prepareSettings();

};

#endif /* SETTINGS_H_ */
