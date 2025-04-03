#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <vector>
#include <Eigen/Dense>
#include "point3D.h"

struct PointCloud
{
    std::vector<Point3D> points;

    // Add a point to the point cloud
    void addPoint(const Point3D &p);

    // Get the size of the point cloud
    size_t size() const;

    // Create a subset from start index to end index
    PointCloud subset(size_t start, size_t end) const;

    // Create a shuffled subset - taking every Nth point
    PointCloud shuffledSubset(size_t startOffset, size_t stride) const;

    // Apply transformation to all points
    void transform(const Eigen::Matrix4d &transformMatrix);
};

#endif // POINTCLOUD_H