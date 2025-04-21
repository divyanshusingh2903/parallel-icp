#ifndef POINT3D_H
#define POINT3D_H

#include <cmath>

// Define point structure
struct Point3D
{
    double x, y, z;

    Point3D();
    Point3D(double x_, double y_, double z_);

    // Calculate Euclidean distance between two points
    double distance(const Point3D &other) const;
};

#endif // POINT3D_H