#include "settings.h"

/***************************************************************************************************
void inputSettings::inputSettings()
private - Default constructor
***************************************************************************************************/
inputSettings::inputSettings(int largc, char** largv)
{
    // Default values for the input parameters
    argc = largc;   //Command line parameters are stored at private var
    argv = largv;   //Command line parameters are stored at private var
    title = "Default Title";
    minfFile = "minf";
    mxyzFile = "mxyz";
    mienFile = "mien";
    mrngFile = "mrng";
    dataFile = "data";
    initT = 0.0;
    D = 1.0;
    source = 0.0;
    nIter = 1;
    dt = 1.0;
    R_1 = 0.01;
}

/***************************************************************************************************
void inputSettings::readSettingsFile()
public - main interface function, calls all other, private, functions of the class
***************************************************************************************************/
void inputSettings::prepareSettings()
{
    readSettingsFile();
    printSettings();

    return;
}


/***************************************************************************************************
void inputSettings::readSettingsFile()
private -reads settings.in file
***************************************************************************************************/
void inputSettings::readSettingsFile()
{
    char * fileName;
    string lineString;
    string dummyString;
    char   dummyChar;

    ifstream inputFile;
    if (argc > 1)
    {
        fileName = argv[1];
    }
    else
    {
        fileName = "settings.in";
    }

    inputFile.open(fileName,ios::in);
    if (inputFile.is_open()==false)
    {
        cout << "Unable to open input file: " << fileName << " ! Aborting... " << endl;
        exit(0);
    }

    while (!inputFile.eof())
    {
        // Get a line and store in lineString
        getline(inputFile, lineString, '\n');
        // If the first character of the line is not a '#'
        if (lineString.c_str()[0] != '#')
        {
            istringstream iss(lineString);
            iss >> dummyString;
            if(dummyString == "title")
                iss >> title;
            else if(dummyString == "minf")
                iss >> minfFile;
            else if(dummyString == "mxyz")
                iss >> mxyzFile;
            else if(dummyString == "mien")
                iss >> mienFile;
            else if(dummyString == "mrng")
                iss >> mrngFile;
            else if(dummyString == "data")
                iss >> dataFile;
            else if(dummyString == "init")
                iss >> initT;
            else if(dummyString == "D")
                iss >> D;
            else if(dummyString == "S")
                iss >> source;
            else if(dummyString == "iter")
                iss >> nIter;
            else if(dummyString == "dt")
                iss >> dt;
            else if(dummyString == "nfg")
            {
                iss >> nFG;
                BC = new bndc [nFG+1];
            }
            else if(dummyString == "fg")
            {
                int fgnumber;
                for (int i=0; i<nFG; i++)
                {
                    iss >> fgnumber;
                    iss >> BC[fgnumber].BCType;
                    iss >> BC[fgnumber].BCValue1;
                    if (BC[fgnumber].BCType == 3)
                        iss >> BC[fgnumber].BCValue2;
                }
            }
            else
            {
                cout << endl << "Unknown keyword in the settings file : " << dummyString;
                cout << endl << "Aborting...";
                inputFile.close();
                exit(0);
            }
        }
    }
    inputFile.close();

    return;
}


/***************************************************************************************************
void inputSettings::printSettings()
***************************************************************************************************/
void inputSettings::printSettings()
{
    cout.precision(4);
    cout << scientific;
        cout << endl << "==================== SETTINGS ====================" << endl;
        cout << "Title of the simualation              : " << title		<< endl;
        cout << "Name of the minf file                 : " << minfFile	<< endl;
        cout << "Name of the mxyz file                 : " << mxyzFile	<< endl;
        cout << "Name of the mien file                 : " << mienFile	<< endl;
        cout << "Name of the mrng file                 : " << mrngFile	<< endl;
        cout << "Name of the initial distribution file : " << dataFile	<< endl;
        cout << "Initial value of the Temperature      : " << initT		<< endl;
        cout << "Diffusion coefficient                 : " << D			<< endl;
        cout << "Source term                           : " << source		<< endl;
        cout << "Number of maximum time steps          : " << nIter		<< endl;
        cout << "Time step size                        : " << dt			<< endl;
        cout << "                                         BCType\tBCValue1\tBCValue2" << endl;
        for (int i=1; i<nFG+1; i++)
            cout << "BC Type and values of face group " << i << "      : "
                << BC[i].BCType << '\t' << BC[i].BCValue1 << '\t' << BC[i].BCValue2 << endl;

    return;
}


