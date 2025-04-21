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

// Add a downsample method to the PointCloud class
PointCloud PointCloud::downsample(double rate) const
{
    if (rate >= 1.0)
    {
        return *this; // Return a copy of the full point cloud
    }

    PointCloud result;
    size_t targetSize = static_cast<size_t>(points.size() * rate);

    // Ensure at least one point
    targetSize = std::max(targetSize, static_cast<size_t>(1));

    // Reservoir sampling algorithm for uniform random sampling
    std::mt19937 gen(42); // Fixed seed for reproducibility

    if (targetSize >= points.size())
    {
        // Just copy all points if we want more than we have
        result.points = points;
    }
    else
    {
        // Initialize with the first targetSize points
        result.points.assign(points.begin(), points.begin() + targetSize);

        // Consider each remaining point for inclusion
        for (size_t i = targetSize; i < points.size(); ++i)
        {
            // Generate a random index in [0, i]
            std::uniform_int_distribution<size_t> dist(0, i);
            size_t j = dist(gen);

            // Replace a point in the reservoir with probability targetSize/i
            if (j < targetSize)
            {
                result.points[j] = points[i];
            }
        }
    }

    return result;
}