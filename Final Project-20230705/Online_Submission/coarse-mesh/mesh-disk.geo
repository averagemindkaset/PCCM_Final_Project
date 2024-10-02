// Gmsh project created on Tue Oct 23 16:34:40 2018
//+
he = 0.005;
rInner = 0.01;
rOuter = 0.1;
Point(1) = {0, 0, 0, he};

// Inner Circle
Point(2) = {rInner, 0, 0, he};
Point(3) = {0, rInner, 0, he};
Point(4) = {-rInner, 0, 0, he};
Point(5) = {0, -rInner, 0, he};

Circle(2) = {2, 1, 3};
Circle(3) = {3, 1, 4};
Circle(4) = {4, 1, 5};
Circle(5) = {5, 1, 2};

// Outer circle
Point(6) = {rOuter, 0, 0, he};
Point(7) = {0, rOuter, 0, he};
Point(8) = {-rOuter, 0, 0, he};
Point(9) = {0, -rOuter, 0, he};

Circle(6) = {6, 1, 7};
Circle(7) = {7, 1, 8};
Circle(8) = {8, 1, 9};
Circle(9) = {9, 1, 6};

// Define surface
Line Loop(1) = {5, 4, 3, 2};
Line Loop(2) = {9, 8, 7, 6};
Plane Surface(1) = {2,1};
Plane Surface(2) = {1};
//+
Physical Line("outer") = {6, 7, 8, 9};
//+
Physical Surface("surface") = {1, 2};
