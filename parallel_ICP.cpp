#include "parallel_ICP.h"

Eigen::Matrix4d ParallelICP::align(PointCloud &source, const PointCloud &target, const Parameters &params)
{
    // Initialize MPI variables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process (rank 0) holds the full transformation
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prevTransformation = Eigen::Matrix4d::Identity();

    // Create local subset of source points using shuffled distribution
    // This improves load balancing by distributing the points randomly
    PointCloud localSource = source.shuffledSubset(rank, size);

    // Save initial state if we're the root process and output file is specified
    if (rank == 0 && !params.outputFilePrefix.empty())
    {
        std::string initialFile = params.outputFilePrefix + "_initial.xyzrgb";
        saveColoredPointClouds(target, source, initialFile);
    }

    // Vectors to store partial sums and correspondence counts
    std::vector<double> localPartialSums(16); // 3x3 cross-covariance matrix + 3D centroid x 2 + count
    std::vector<double> allPartialSums;
    std::vector<int> allCorrespondenceCounts;

    if (rank == 0)
    {
        allPartialSums.resize(16 * size);
        allCorrespondenceCounts.resize(size);
    }

    double error = std::numeric_limits<double>::max();
    double prevError = std::numeric_limits<double>::max();

    // Main ICP loop
    for (int iteration = 0; iteration < params.maxIterations; ++iteration)
    {
        // Each process finds correspondences and computes partial sums
        int localCorrespondenceCount = 0;
        localPartialSums = findPartialSums(localSource, target, params.outlierRejectionThreshold, localCorrespondenceCount);

        // Gather all partial sums and correspondence counts to root process
        MPI_Gather(localPartialSums.data(), 16, MPI_DOUBLE,
                   allPartialSums.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Gather(&localCorrespondenceCount, 1, MPI_INT,
                   allCorrespondenceCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Root process calculates transformation
        if (rank == 0)
        {
            // Store previous transformation for convergence check
            prevTransformation = transformation;

            // Calculate new transformation from partial sums
            Eigen::Matrix4d incrementalTransform = calculateTransformationFromPartialSums(
                allPartialSums, allCorrespondenceCounts, size);

            // Update cumulative transformation
            transformation = incrementalTransform * transformation;

            // Apply transformation to the full source cloud for visualization
            PointCloud transformedSource = source;
            transformedSource.transform(transformation);

            // Calculate error (convergence metric)
            std::vector<std::pair<Point3D, Point3D>> dummyCorrespondences;
            error = findCorrespondences(transformedSource, target, dummyCorrespondences, params.outlierRejectionThreshold);

            // Save intermediate result if requested
            if (!params.outputFilePrefix.empty() && iteration % params.saveInterval == 0)
            {
                std::string iterFile = params.outputFilePrefix + "_iter" + std::to_string(iteration) + ".xyzrgb";
                saveColoredPointClouds(target, transformedSource, iterFile);
            }

            // Check for convergence
            double transformDiff = (prevTransformation.inverse() * transformation - Eigen::Matrix4d::Identity()).norm();
            if (transformDiff < params.convergenceThreshold)
            {
                std::cout << "Converged after " << iteration + 1 << " iterations." << std::endl;
                break;
            }

            std::cout << "Iteration " << iteration + 1 << ", Error: " << error
                      << ", Change: " << std::abs(prevError - error) << std::endl;
            prevError = error;
        }

        // Broadcast new transformation to all processes
        MPI_Bcast(transformation.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Update local source points with the new transformation
        localSource.transform(transformation);
    }

    // Save final result if we're the root process
    if (rank == 0 && !params.outputFilePrefix.empty())
    {
        PointCloud transformedSource = source;
        transformedSource.transform(transformation);

        std::string finalFile = params.outputFilePrefix + "_final.xyzrgb";
        saveColoredPointClouds(target, transformedSource, finalFile);

        // Save only the aligned source points
        std::ofstream alignedFile(params.outputFilePrefix + "_aligned.xyz");
        for (const auto &p : transformedSource.points)
        {
            alignedFile << p.x << " " << p.y << " " << p.z << " 0 255 0" << std::endl;
        }
        alignedFile.close();
    }

    return transformation;
}

std::vector<double> ParallelICP::findPartialSums(const PointCloud &localSource, const PointCloud &target,
                                                 double outlierThreshold, int &numCorrespondences)
{
    // Vector to store partial sums (9 for cross-covariance, 3 for source centroid, 3 for target centroid)
    std::vector<double> partialSums(16, 0.0);
    numCorrespondences = 0;

    // Centroids for current process
    Point3D sourceSum = {0.0, 0.0, 0.0};
    Point3D targetSum = {0.0, 0.0, 0.0};

    // Cross-covariance matrix elements (stored as 9 values)
    std::vector<double> covMatrix(9, 0.0);

    // Find correspondences and compute sums
    for (const auto &sourcePoint : localSource.points)
    {
        // Find closest point in target
        double minDist = std::numeric_limits<double>::max();
        Point3D closestPoint;

        for (const auto &targetPoint : target.points)
        {
            double dist = sourcePoint.distance(targetPoint);
            if (dist < minDist)
            {
                minDist = dist;
                closestPoint = targetPoint;
            }
        }

        // Apply outlier rejection if threshold is set
        if (outlierThreshold > 0 && minDist > outlierThreshold)
        {
            continue; // Skip this correspondence
        }

        // Update sums
        sourceSum.x += sourcePoint.x;
        sourceSum.y += sourcePoint.y;
        sourceSum.z += sourcePoint.z;

        targetSum.x += closestPoint.x;
        targetSum.y += closestPoint.y;
        targetSum.z += closestPoint.z;

        // Update cross-covariance terms (using outer product)
        covMatrix[0] += sourcePoint.x * closestPoint.x; // Sxx
        covMatrix[1] += sourcePoint.x * closestPoint.y; // Sxy
        covMatrix[2] += sourcePoint.x * closestPoint.z; // Sxz
        covMatrix[3] += sourcePoint.y * closestPoint.x; // Syx
        covMatrix[4] += sourcePoint.y * closestPoint.y; // Syy
        covMatrix[5] += sourcePoint.y * closestPoint.z; // Syz
        covMatrix[6] += sourcePoint.z * closestPoint.x; // Szx
        covMatrix[7] += sourcePoint.z * closestPoint.y; // Szy
        covMatrix[8] += sourcePoint.z * closestPoint.z; // Szz

        numCorrespondences++;
    }

    // Store results in partialSums vector
    for (int i = 0; i < 9; i++)
    {
        partialSums[i] = covMatrix[i];
    }

    partialSums[9] = sourceSum.x;
    partialSums[10] = sourceSum.y;
    partialSums[11] = sourceSum.z;
    partialSums[12] = targetSum.x;
    partialSums[13] = targetSum.y;
    partialSums[14] = targetSum.z;
    partialSums[15] = numCorrespondences;

    return partialSums;
}

Eigen::Matrix4d ParallelICP::calculateTransformationFromPartialSums(const std::vector<double> &allPartialSums,
                                                                    const std::vector<int> &numCorrespondences,
                                                                    int numProcesses)
{
    // Total number of correspondences
    int totalCorrespondences = 0;
    for (int i = 0; i < numProcesses; i++)
    {
        totalCorrespondences += numCorrespondences[i];
    }

    if (totalCorrespondences == 0)
    {
        std::cerr << "Error: No valid correspondences found." << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    // Accumulate centroids and covariance matrix
    Eigen::Vector3d sourceCentroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetCentroid = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covMatrix = Eigen::Matrix3d::Zero();

    for (int i = 0; i < numProcesses; i++)
    {
        // Extract partial sums from this process
        const double *processSums = &allPartialSums[i * 16];
        int processCorrespondences = numCorrespondences[i];

        if (processCorrespondences == 0)
            continue;

        // Add to covariance matrix
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                covMatrix(j, k) += processSums[j * 3 + k];
            }
        }

        // Add to centroids
        sourceCentroid[0] += processSums[9];
        sourceCentroid[1] += processSums[10];
        sourceCentroid[2] += processSums[11];

        targetCentroid[0] += processSums[12];
        targetCentroid[1] += processSums[13];
        targetCentroid[2] += processSums[14];
    }

    // Calculate final centroids
    sourceCentroid /= totalCorrespondences;
    targetCentroid /= totalCorrespondences;

    // Adjust covariance matrix for centroids (Equation 5 from the paper)
    Eigen::Matrix3d adjustedCovMatrix = covMatrix / totalCorrespondences -
                                        sourceCentroid * targetCentroid.transpose();

    // Use SVD to compute optimal rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(adjustedCovMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();

    // Ensure we have a proper rotation matrix (det=1)
    if (rotation.determinant() < 0)
    {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        rotation = V * svd.matrixU().transpose();
    }

    // Compute translation
    Eigen::Vector3d translation = targetCentroid - rotation * sourceCentroid;

    // Build 4x4 transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;

    return transform;
}

double ParallelICP::findCorrespondences(const PointCloud &source, const PointCloud &target, std::vector<std::pair<Point3D, Point3D>> &correspondences, double outlierThreshold)
{
    correspondences.clear();
    double totalError = 0.0;
    int numCorrespondences = 0;

    for (const auto &sourcePoint : source.points)
    {
        // Find closest point in target
        double minDist = std::numeric_limits<double>::max();
        Point3D closestPoint;

        for (const auto &targetPoint : target.points)
        {
            double dist = sourcePoint.distance(targetPoint);
            if (dist < minDist)
            {
                minDist = dist;
                closestPoint = targetPoint;
            }
        }

        // Apply outlier rejection if threshold is set
        if (outlierThreshold > 0 && minDist > outlierThreshold)
        {
            continue; // Skip this correspondence
        }

        correspondences.push_back(std::make_pair(sourcePoint, closestPoint));
        totalError += minDist * minDist;
        numCorrespondences++;
    }

    return (numCorrespondences > 0) ? (totalError / numCorrespondences) : std::numeric_limits<double>::max();
}

void ParallelICP::saveColoredPointClouds(const PointCloud &target, const PointCloud &source, const std::string &filename)
{
    std::ofstream file(filename);

    // Save target points in blue
    for (const auto &p : target.points)
    {
        file << p.x << " " << p.y << " " << p.z << " 0 0 255" << std::endl;
    }

    // Save source points in red
    for (const auto &p : source.points)
    {
        file << p.x << " " << p.y << " " << p.z << " 255 0 0" << std::endl;
    }

    file.close();
}