#include "parallel_ICP.h"

Eigen::Matrix4d ParallelICP::align(PointCloud &source, const PointCloud &target, const Parameters &params)
{
    // Initialize MPI variables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process (rank 0) holds the full transformation
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();

    // Save initial state if output prefix is provided
    if (rank == 0 && !params.outputFilePrefix.empty())
    {
        saveColoredPointClouds(target, source, params.outputFilePrefix + "_initial.xyzrgb");
    }

    // Define multi-resolution levels
    const std::vector<double> resolutionLevels = {0.05, 0.1, 0.25, 0.5, 1.0};
    const int numLevels = resolutionLevels.size();

    // Calculate iterations per level - divide equally among all levels
    int totalIterations = params.maxIterations;
    std::vector<int> iterationsPerLevel(numLevels);

    // Divide iterations equally
    int baseIterationsPerLevel = totalIterations / numLevels;

    // Initialize all levels with the base number of iterations
    for (int i = 0; i < numLevels; ++i)
    {
        iterationsPerLevel[i] = baseIterationsPerLevel;
    }

    // Add remaining iterations to the highest resolution level (last level)
    int remainingIterations = totalIterations % numLevels;
    iterationsPerLevel[numLevels - 1] += remainingIterations;

    // Keep the original point clouds
    PointCloud originalSource = source;
    PointCloud originalTarget = target;

    // Process each resolution level
    for (int level = 0; level < numLevels; ++level)
    {
        if (rank == 0)
        {
            std::cout << "\n===== Resolution Level " << level + 1 << "/" << numLevels
                      << " (" << resolutionLevels[level] * 100 << "%, "
                      << iterationsPerLevel[level] << " iterations) =====" << std::endl;
        }

        // Downsample the point clouds for this level
        PointCloud levelSource = originalSource.downsample(resolutionLevels[level]);
        PointCloud levelTarget = originalTarget.downsample(resolutionLevels[level]);

        // Apply the current transformation to the source
        levelSource.transform(transformation);

        // Parameters for this level
        Parameters levelParams = params;
        levelParams.maxIterations = iterationsPerLevel[level];

        // Adjust convergence threshold based on resolution level
        if (level < numLevels - 1)
        {
            // Use larger threshold for coarser levels
            levelParams.convergenceThreshold = params.convergenceThreshold *
                                               (1.0 + 2.0 * (numLevels - level - 1) / numLevels);
        }

        // Create local subset of source points using shuffled distribution
        PointCloud localSource = levelSource.shuffledSubset(rank, size);
        PointCloud localSourceTransformed = localSource; // Keep a separate copy for transformation

        // Vectors to store partial sums and correspondence counts for transform calculation
        std::vector<double> localPartialSums_transform(16);
        std::vector<double> allPartialSums_transform;
        std::vector<int> allCorrespondenceCounts_transform;

        if (rank == 0)
        {
            allPartialSums_transform.resize(16 * size);
            allCorrespondenceCounts_transform.resize(size);
        }

        double error = std::numeric_limits<double>::max();
        double prevError = std::numeric_limits<double>::max();
        Eigen::Matrix4d prevTransformation = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d levelTransformation = Eigen::Matrix4d::Identity();

        // ICP loop for this resolution level
        for (int iteration = 0; iteration < levelParams.maxIterations; ++iteration)
        {
            // 1. Each process finds correspondences and computes PARTIAL SUMS FOR TRANSFORMATION
            int localCorrespondenceCount_transform = 0;
            localPartialSums_transform = findPartialSums(localSourceTransformed, levelTarget,
                                                         levelParams.outlierRejectionThreshold,
                                                         localCorrespondenceCount_transform);

            // 2. Gather partial sums and counts for TRANSFORMATION to root process
            MPI_Gather(localPartialSums_transform.data(), 16, MPI_DOUBLE,
                       allPartialSums_transform.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            MPI_Gather(&localCorrespondenceCount_transform, 1, MPI_INT,
                       allCorrespondenceCounts_transform.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

            Eigen::Matrix4d incrementalTransform = Eigen::Matrix4d::Identity();

            // 3. Root process calculates incremental transformation
            if (rank == 0)
            {
                prevTransformation = levelTransformation;

                // Calculate new incremental transformation from partial sums
                incrementalTransform = calculateTransformationFromPartialSums(
                    allPartialSums_transform, allCorrespondenceCounts_transform, size);

                // Update level transformation
                levelTransformation = incrementalTransform * levelTransformation;
            }

            // 4. Broadcast the NEW CUMULATIVE transformation to all processes
            MPI_Bcast(levelTransformation.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // 5. Update local source points with the new transformation
            localSourceTransformed = localSource;
            localSourceTransformed.transform(levelTransformation);

            // 6. PARALLEL ERROR CALCULATION
            PartialErrorResult localErrorResult = calculatePartialErrorSum(
                localSourceTransformed, levelTarget, levelParams.outlierRejectionThreshold);

            // 7. Reduce partial errors and counts to root process
            double totalErrorSum = 0.0;
            int totalCorrespondenceCount_error = 0;

            MPI_Reduce(&localErrorResult.sumSquaredError, &totalErrorSum, 1,
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localErrorResult.numCorrespondences, &totalCorrespondenceCount_error, 1,
                       MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            // 8. Root process calculates final MSE and checks convergence
            if (rank == 0)
            {
                // Calculate final MSE
                prevError = error;
                error = (totalCorrespondenceCount_error > 0) ? (totalErrorSum / totalCorrespondenceCount_error) : std::numeric_limits<double>::max();

                // Calculate difference between previous and current transformation
                double transformDiff = (prevTransformation.inverse() * levelTransformation -
                                        Eigen::Matrix4d::Identity())
                                           .norm();

                std::cout << "Level " << level + 1 << ", Iteration " << iteration + 1
                          << ", Points: " << levelSource.points.size()
                          << ", Error: " << error
                          << ", Transform Diff: " << transformDiff << std::endl;

                // Save intermediate result if requested
                if (!levelParams.outputFilePrefix.empty() &&
                    (iteration == levelParams.maxIterations - 1 ||
                     iteration % levelParams.saveInterval == 0))
                {
                    PointCloud transformedSource = originalSource;
                    transformedSource.transform(transformation * levelTransformation);
                    std::string iterFile = levelParams.outputFilePrefix + "_level" +
                                           std::to_string(level + 1) + "_iter" +
                                           std::to_string(iteration + 1) + ".xyzrgb";
                    saveColoredPointClouds(originalTarget, transformedSource, iterFile);
                }

                if (transformDiff < levelParams.convergenceThreshold)
                {
                    std::cout << "Level " << level + 1 << " converged after " << iteration + 1
                              << " iterations (threshold: " << levelParams.convergenceThreshold << ")" << std::endl;
                    int convergence_flag = 1;
                    MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    break;
                }

                // Broadcast non-convergence flag
                int convergence_flag = 0;
                MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
            else
            {
                // Worker processes check the convergence flag
                int convergence_flag = 0;
                MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (convergence_flag == 1)
                {
                    break;
                }
            }
        } // End of level ICP loop

        // Update the cumulative transformation
        if (rank == 0)
        {
            transformation = levelTransformation * transformation;
        }

        // Broadcast the updated cumulative transformation to all processes
        MPI_Bcast(transformation.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } // End of resolution levels loop

    // Save final result
    if (rank == 0 && !params.outputFilePrefix.empty())
    {
        PointCloud transformedSourceFinal = originalSource;
        transformedSourceFinal.transform(transformation);
        saveColoredPointClouds(originalTarget, transformedSourceFinal,
                               params.outputFilePrefix + "_final.xyzrgb");
    }

    // Update the source point cloud with the final transformation
    if (rank == 0)
    {
        source.transform(transformation);
    }

    return transformation;
}

std::vector<double> ParallelICP::findPartialSums(const PointCloud &localSource, const PointCloud &target, double outlierThreshold, int &numCorrespondences)
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

Eigen::Matrix4d ParallelICP::calculateTransformationFromPartialSums(const std::vector<double> &allPartialSums, const std::vector<int> &numCorrespondences, int numProcesses)
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

ParallelICP::PartialErrorResult ParallelICP::calculatePartialErrorSum(const PointCloud &localSource, const PointCloud &target, double outlierThreshold)
{
    PartialErrorResult result;

    for (const auto &sourcePoint : localSource.points)
    {
        // Find closest point in target
        double minDistSq = std::numeric_limits<double>::max(); // Use squared distance
        Point3D closestPoint;                                  // Keep track of the actual point if needed for debugging, otherwise not strictly necessary here

        for (const auto &targetPoint : target.points)
        {
            // Calculate squared distance directly if possible to avoid sqrt
            double dx = sourcePoint.x - targetPoint.x;
            double dy = sourcePoint.y - targetPoint.y;
            double dz = sourcePoint.z - targetPoint.z;
            double distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < minDistSq)
            {
                minDistSq = distSq;
                // closestPoint = targetPoint; // Assign if needed
            }
        }

        // Apply outlier rejection if threshold is set (compare squared distances)
        // Note: Ensure outlierThreshold is squared if it represents a distance
        double outlierThresholdSq = (outlierThreshold > 0) ? outlierThreshold * outlierThreshold : -1.0;
        if (outlierThresholdSq > 0 && minDistSq > outlierThresholdSq)
        {
            continue; // Skip this correspondence
        }

        // Ensure minDistSq is not infinity before adding
        if (minDistSq != std::numeric_limits<double>::max())
        {
            result.sumSquaredError += minDistSq; // Add squared distance
            result.numCorrespondences++;
        }
    }

    return result;
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
