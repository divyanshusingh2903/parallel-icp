#include "point3D.h"

// Default constructor
Point3D::Point3D() : x(0), y(0), z(0) {}

// Parameterized constructor
Point3D::Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

// Calculate Euclidean distance between two points
double Point3D::distance(const Point3D &other) const
{
    return std::sqrt(std::pow(x - other.x, 2) +
                     std::pow(y - other.y, 2) +
                     std::pow(z - other.z, 2));
}