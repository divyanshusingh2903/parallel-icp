#ifndef PARALLEL_ICP_H
#define PARALLEL_ICP_H

#include "pointCloud.h"
#include <mpi.h>
#include <Eigen/SVD>
#include <limits>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>

class ParallelICP
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
     * Run Parallel ICP algorithm to align source cloud with target cloud
     * @param source Source point cloud to be aligned
     * @param target Target point cloud (reference)
     * @param params Algorithm parameters
     * @return Transformation matrix that aligns source with target (valid only on root process)
     */
    static Eigen::Matrix4d align(PointCloud &source, const PointCloud &target, const Parameters &params);

private:
    /**
     * Find closest point in target for each point in source (distributed processing)
     * @param localSource Local portion of source point cloud
     * @param target Target point cloud
     * @param localCorrespondences Output vector of corresponding point pairs
     * @param outlierThreshold Threshold for rejecting outlier matches (-1 means no rejection)
     * @return Partial sums for calculating cross-covariance matrix
     */
    static std::vector<double> findPartialSums(const PointCloud &localSource, const PointCloud &target, double outlierThreshold, int &numCorrespondences);

    /**
     * Calculate optimal transformation using partial sums from all processes
     * @param partialSums Array of partial sums from all processes
     * @param numCorrespondences Array of correspondence counts from all processes
     * @param numProcesses Number of processes
     * @return Transformation matrix (4x4)
     */
    static Eigen::Matrix4d calculateTransformationFromPartialSums(const std::vector<double> &partialSums, const std::vector<int> &numCorrespondences, int numProcesses);

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
     * Save point clouds to XYZ file with colors
     * @param target Target point cloud (blue)
     * @param source Source point cloud (red)
     * @param filename Output filename
     */
    static void saveColoredPointClouds(const PointCloud &target, const PointCloud &source, const std::string &filename);
};

#endif // PARALLEL_ICP_H