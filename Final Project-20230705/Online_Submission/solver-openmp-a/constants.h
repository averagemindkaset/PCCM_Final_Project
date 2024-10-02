#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <limits>
#include <time.h>

using namespace std;

const double PI = 3.14159265;   // Pi constant

const int nGQP = 7;             // Number of Gauss quadrature points

const int nsd = 2;  // number of space dimensions
const int nen = 3;  // number of element nodes
const int nef = 3;  // number of element faces
const int xsd = 0;
const int ysd = 1;

const int edgeNodes[3][2] = {{0,1},{1,2},{2,0}};

#endif /* CONSTANTS_H_ */
