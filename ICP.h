// ICP.h
#ifndef ICP_H
#define ICP_H

#include "pointCloud.h"
#include <Eigen/SVD>
#include <limits>
#include <iostream>
#include <string>

class ICP
{
public:
    // Parameters for ICP algorithm
    struct Parameters
    {
        double convergenceThreshold = 1e-5;    // Convergence threshold for transformation
        int maxIterations = 10;                // Maximum number of iterations
        double outlierRejectionThreshold = -1; // Threshold for rejecting outlier correspondences (-1 means no rejection)
        std::string outputFilePrefix = "";     // Prefix for intermediate output files (empty means no intermediate files)
        int saveInterval = 1;                  // Save every N iterations

        // Default constructor
        Parameters() : convergenceThreshold(1e-5), maxIterations(50), outlierRejectionThreshold(-1),
                       outputFilePrefix(""), saveInterval(5) {}
    };

    /**
     * Run ICP algorithm to align source cloud with target cloud
     * @param source Source point cloud to be aligned
     * @param target Target point cloud (reference)
     * @param params Algorithm parameters
     * @return Transformation matrix that aligns source with target
     */
    static Eigen::Matrix4d align(const PointCloud &source, const PointCloud &target, const Parameters &params);

private:
    /**
     * Find closest point in target for each point in source
     * @param source Transformed source point cloud
     * @param target Target point cloud
     * @param correspondences Output vector of corresponding point pairs
     * @param outlierThreshold Threshold for rejecting outlier matches (-1 means no rejection)
     * @return Mean squared error between corresponding points
     */
    static double findCorrespondences(const PointCloud &source, const PointCloud &target, std::vector<std::pair<Point3D, Point3D>> &correspondences, double outlierThreshold);

    /**
     * Calculate optimal transformation that aligns source points with their corresponding target points
     * @param correspondences Vector of corresponding point pairs
     * @return Transformation matrix (4x4)
     */
    static Eigen::Matrix4d calculateTransformation(const std::vector<std::pair<Point3D, Point3D>> &correspondences);

    /**
     * Save point clouds to XYZ file with colors
     * @param target Target point cloud (blue)
     * @param source Source point cloud (red)
     * @param filename Output filename
     */
    static void saveColoredPointClouds(const PointCloud &target, const PointCloud &source, const std::string &filename);
};

#endif // ICP_H