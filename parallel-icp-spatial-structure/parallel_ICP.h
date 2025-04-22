#ifndef PARALLEL_ICP
#define PARALLEL_ICP

#include "../point-cloud/pointCloud.h"
#include <mpi.h>
#include <Eigen/SVD>
#include <limits>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>
#include <memory>
#include <nanoflann.hpp>

// KD-tree adapter for PointCloud
struct KDTreeAdapter
{
    const PointCloud &pc;

    // Constructor
    KDTreeAdapter(const PointCloud &point_cloud) : pc(point_cloud) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pc.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class
    inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t /*size*/) const
    {
        const double d0 = p1[0] - pc.points[idx_p2].x;
        const double d1 = p1[1] - pc.points[idx_p2].y;
        const double d2 = p1[2] - pc.points[idx_p2].z;
        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    // Returns the dim'th component of the idx'th point in the class
    inline double kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0)
            return pc.points[idx].x;
        else if (dim == 1)
            return pc.points[idx].y;
        else
            return pc.points[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
};

// Define the KD-tree type
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KDTreeAdapter>, KDTreeAdapter, 3> KDTree; /* 3 is dimensionality of points */

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
        double approxNNEpsilon = 0.0;          // Epsilon parameter for approximate nearest neighbor (0.0 for exact search)
        int leafSize = 10;                     // Leaf size for KD-tree

        // Default constructor
        Parameters() : convergenceThreshold(1e-5), maxIterations(50), outlierRejectionThreshold(-1),
                       outputFilePrefix(""), saveInterval(5), approxNNEpsilon(0.0), leafSize(10) {}
    };

    // Structure to hold the result of partial error calculation
    struct PartialErrorResult
    {
        double sumSquaredError = 0.0;
        int numCorrespondences = 0;
    };

    /**
     * @brief Aligns two point clouds using the ICP algorithm with KD-tree optimization.
     *
     * The function takes two point clouds as input and uses a KD-tree to accelerate
     * the nearest neighbor search during correspondence finding.
     *
     * @param source The source point cloud to be aligned.
     * @param target The target point cloud to which the source point cloud is to be aligned.
     * @param params Parameters for the ICP algorithm.
     *
     * @return The final transformation matrix that aligns the source point cloud to the target point cloud.
     */
    static Eigen::Matrix4d align(PointCloud &source, const PointCloud &target, const Parameters &params);

private:
    /**
     * Find closest point in target for each point in source using KD-tree (distributed processing)
     * @param localSource Local portion of source point cloud
     * @param target Target point cloud
     * @param kdtree KD-tree built from target point cloud
     * @param outlierThreshold Threshold for rejecting outlier matches (-1 means no rejection)
     * @param approxEpsilon Epsilon parameter for approximate nearest neighbor (0.0 for exact search)
     * @param numCorrespondences Output number of valid correspondences
     * @return Partial sums for calculating cross-covariance matrix
     */
    static std::vector<double> findPartialSumsWithKDTree(const PointCloud &localSource, const PointCloud &target, const KDTree &kdtree, double outlierThreshold, double approxEpsilon, int &numCorrespondences);

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
     * @param kdtree KD-tree built from target point cloud
     * @param outlierThreshold Distance threshold for rejecting outlier correspondences
     * @param approxEpsilon Epsilon parameter for approximate nearest neighbor
     * @return PartialErrorResult containing the sum of squared errors and the number of correspondences
     */
    static PartialErrorResult calculatePartialErrorSumWithKDTree(const PointCloud &localSource, const PointCloud &target, const KDTree &kdtree, double outlierThreshold, double approxEpsilon);
};

#endif // PARALLEL_ICP