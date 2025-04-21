#ifndef PARALLEL_ICP_H
#define PARALLEL_ICP_H

#include "../point-cloud/pointCloud.h"
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

    // Structure to hold the result of partial error calculation
    struct PartialErrorResult
    {
        double sumSquaredError = 0.0;
        int numCorrespondences = 0;
    };

    /**
     * @brief Aligns two point clouds using the ICP algorithm, with a multi-resolution
     *        approach.
     *
     * The function takes two point clouds as input, and uses a multi-resolution approach
     * to align them. The point clouds are downsampled to a series of resolutions, and the
     * ICP algorithm is run at each resolution level, with the transformation calculated
     * at each level being used as the initial guess for the next level. The final
     * transformation is then applied to the source point cloud.
     *
     * @param source The source point cloud to be aligned.
     * @param target The target point cloud to which the source point cloud is to be
     *               aligned.
     * @param params A structure containing parameters for the ICP algorithm. These
     *               include the maximum number of iterations, the outlier rejection
     *               threshold, the convergence threshold, the save interval, and the
     *               output file prefix.
     *
     * @return The final transformation matrix that aligns the source point cloud to the
     *         target point cloud.
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
     * Save point clouds to XYZ file with colors
     * @param target Target point cloud (blue)
     * @param source Source point cloud (red)
     * @param filename Output filename
     */
    static void saveColoredPointClouds(const PointCloud &target, const PointCloud &source, const std::string &filename);

    /**
     * Calculates the partial sum of squared errors for a subset of source points against a target point cloud.
     * @param localSource Subset of source points to calculate error for (already transformed)
     * @param target Target point cloud to find closest points in
     * @param outlierThreshold Distance threshold for rejecting outlier correspondences
     * @return PartialErrorResult containing the sum of squared errors and the number of correspondences
     */
    static PartialErrorResult calculatePartialErrorSum(const PointCloud &localSource, const PointCloud &target, double outlierThreshold);
};

#endif // PARALLEL_ICP_H