#include "pointCloud.h"


void PointCloud::addPoint(const Point3D &p)
{
    points.push_back(p);
}

size_t PointCloud::size() const
{
    return points.size();
}

// Create a subset from start index to end index
PointCloud PointCloud::subset(size_t start, size_t end) const
{
    PointCloud result;
    for (size_t i = start; i < end && i < points.size(); i++)
    {
        result.addPoint(points[i]);
    }
    return result;
}

// Create a shuffled subset - taking every Nth point
PointCloud PointCloud::shuffledSubset(size_t startOffset, size_t stride) const
{
    PointCloud result;
    for (size_t i = startOffset; i < points.size(); i += stride)
    {
        result.addPoint(points[i]);
    }
    return result;
}

// Apply transformation to all points
void PointCloud::transform(const Eigen::Matrix4d &transformMatrix)
{
    for (auto &point : points)
    {
        // Convert to homogeneous coordinates
        Eigen::Vector4d p(point.x, point.y, point.z, 1.0);

        // Apply transformation
        Eigen::Vector4d p_transformed = transformMatrix * p;

        // Convert back
        point.x = p_transformed(0);
        point.y = p_transformed(1);
        point.z = p_transformed(2);
    }
}
