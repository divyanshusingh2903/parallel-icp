#include "parallel_ICP.h"

Eigen::Matrix4d ParallelICP::align(PointCloud &source, const PointCloud &target, const Parameters &params)
{
    // Save initial state if output prefix is provided
    if (!params.outputFilePrefix.empty())
    {
        saveColoredPointClouds(target, source, params.outputFilePrefix + "_initial.xyzrgb");
    }

    // Initialize MPI variables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process (rank 0) holds the full transformation
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d prevTransformation = Eigen::Matrix4d::Identity(); // Only needed on rank 0 for convergence

    // Create local subset of source points using shuffled distribution
    PointCloud localSource = source.shuffledSubset(rank, size);
    PointCloud localSourceTransformed = localSource; // Keep a separate copy for transformation

    // --- KD-Tree Optimization ---
    // Build KD-tree for target point cloud once, broadcast a flag when it's ready
    // The KD-tree is only constructed on rank 0 and then used by all processes
    KDTreeAdapter kdtree_adapter(target);
    KDTree kdtree(3, kdtree_adapter, nanoflann::KDTreeSingleIndexAdaptorParams(params.leafSize));

    // Root process builds the KD-tree
    if (rank == 0)
    {
        kdtree.buildIndex();
        std::cout << "KD-tree built with " << target.size() << " points." << std::endl;
    }

    // Make sure all processes wait until KD-tree is ready
    MPI_Barrier(MPI_COMM_WORLD);

    // Vectors to store partial sums and correspondence counts for transform calculation
    std::vector<double> localPartialSums_transform(16); // Renamed to avoid confusion
    std::vector<double> allPartialSums_transform;
    std::vector<int> allCorrespondenceCounts_transform;

    if (rank == 0)
    {
        allPartialSums_transform.resize(16 * size);
        allCorrespondenceCounts_transform.resize(size);
    }

    double error = std::numeric_limits<double>::max();     // Error metric (MSE) - only relevant on rank 0
    double prevError = std::numeric_limits<double>::max(); // Previous error - only relevant on rank 0

    // Main ICP loop
    for (int iteration = 0; iteration < params.maxIterations; ++iteration)
    {
        // 1. Each process finds correspondences and computes PARTIAL SUMS FOR TRANSFORMATION using KD-tree
        int localCorrespondenceCount_transform = 0; // Count for transform calculation
        localPartialSums_transform = findPartialSumsWithKDTree(
            localSourceTransformed,
            target,
            kdtree,
            params.outlierRejectionThreshold,
            params.approxNNEpsilon, // Use epsilon parameter for approximate NN
            localCorrespondenceCount_transform);

        // 2. Gather partial sums and counts for TRANSFORMATION to root process
        MPI_Gather(localPartialSums_transform.data(), 16, MPI_DOUBLE,
                   allPartialSums_transform.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Gather(&localCorrespondenceCount_transform, 1, MPI_INT,
                   allCorrespondenceCounts_transform.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        Eigen::Matrix4d incrementalTransform = Eigen::Matrix4d::Identity(); // Initialize outside rank 0 scope

        // 3. Root process calculates incremental transformation
        if (rank == 0)
        {
            prevTransformation = transformation; // Store previous full transformation

            // Calculate new incremental transformation from partial sums
            incrementalTransform = calculateTransformationFromPartialSums(
                allPartialSums_transform, allCorrespondenceCounts_transform, size);

            // Update cumulative transformation
            transformation = incrementalTransform * transformation;
        }

        // 4. Broadcast the NEW CUMULATIVE transformation to all processes
        MPI_Bcast(transformation.data(), 16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 5. Update local source points with the new CUMULATIVE transformation for the NEXT iteration's partial sums AND for error calculation
        localSourceTransformed = localSource;             // Start from original local subset
        localSourceTransformed.transform(transformation); // Apply full current transform

        // 6. PARALLEL ERROR CALCULATION using KD-tree
        PartialErrorResult localErrorResult = calculatePartialErrorSumWithKDTree(
            localSourceTransformed,
            target,
            kdtree,
            params.outlierRejectionThreshold,
            params.approxNNEpsilon);

        // 7. Reduce partial errors and counts to root process
        double totalErrorSum = 0.0;
        int totalCorrespondenceCount_error = 0; // Use a separate count for error calculation

        MPI_Reduce(&localErrorResult.sumSquaredError, &totalErrorSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localErrorResult.numCorrespondences, &totalCorrespondenceCount_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // 8. Root process calculates final MSE and checks convergence
        if (rank == 0)
        {
            // Calculate final MSE
            prevError = error; // Store previous error
            error = (totalCorrespondenceCount_error > 0) ? (totalErrorSum / totalCorrespondenceCount_error) : std::numeric_limits<double>::max();

            // --- Convergence Check ---
            // Calculate difference between previous and current cumulative transformation
            double transformDiff = (prevTransformation.inverse() * transformation - Eigen::Matrix4d::Identity()).norm();

            std::cout << "Iteration " << iteration + 1 << ", Error: " << error
                      << ", Change: " << std::abs(prevError - error)
                      << ", Transform Diff: " << transformDiff << std::endl;

            // Save intermediate result if requested
            if (!params.outputFilePrefix.empty() && iteration % params.saveInterval == 0)
            {
                PointCloud transformedSourceFull = source;       // Need original full source
                transformedSourceFull.transform(transformation); // Apply current full transform
                std::string iterFile = params.outputFilePrefix + "_iter" + std::to_string(iteration + 1) + ".xyzrgb";
                saveColoredPointClouds(target, transformedSourceFull, iterFile);
            }

            if (transformDiff < params.convergenceThreshold)
            {
                std::cout << "Converged after " << iteration + 1 << " iterations based on transform difference." << std::endl;
                int convergence_flag = 1;
                MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                break;
            }

            // Broadcast non-convergence flag if not breaking
            int convergence_flag = 0;
            MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else
        {
            // Worker processes check the convergence flag broadcasted by root
            int convergence_flag = 0;
            MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (convergence_flag == 1)
            {
                break; // Exit loop if root signaled convergence
            }
        }
    } // End of main ICP loop

    // --- Final Saving (Rank 0) ---
    if (rank == 0 && !params.outputFilePrefix.empty())
    {
        PointCloud transformedSourceFinal = source;       // Use original full source cloud
        transformedSourceFinal.transform(transformation); // Apply final transformation

        // Save only the aligned source points
        std::ofstream alignedFile(params.outputFilePrefix + "_aligned.xyzrgb");
        if (alignedFile.is_open())
        { // Check if file opened successfully
            for (const auto &p : transformedSourceFinal.points)
            {
                // Assuming Point3D has members x, y, z
                alignedFile << p.x << " " << p.y << " " << p.z << " 0 255 0" << std::endl;
            }
            alignedFile.close();
        }
        else
        {
            std::cerr << "Error: Could not open file " << params.outputFilePrefix + "_aligned.xyz" << " for writing." << std::endl;
        }
    }

    // Ensure all processes return the final transformation
    return transformation;
}

// Find correspondences using KD-tree acceleration
std::vector<double> ParallelICP::findPartialSumsWithKDTree(const PointCloud &localSource, const PointCloud &target, const KDTree &kdtree, double outlierThreshold, double approxEpsilon, int &numCorrespondences)
{
    std::vector<double> partialSums(16, 0.0); // For cross-covariance and centroids

    // Initialize counters and sum accumulators
    Eigen::Vector3d sum_source(0, 0, 0);
    Eigen::Vector3d sum_target(0, 0, 0);
    numCorrespondences = 0;

    // For each point in local source, find closest point in target using KD-tree
    for (const auto &sourcePoint : localSource.points)
    {
        // Setup query point
        double query_pt[3] = {sourcePoint.x, sourcePoint.y, sourcePoint.z};

        // Create result set for single nearest neighbor search
        size_t ret_index;
        double out_dist_sqr;

        // Find nearest neighbor - can use approximate nearest neighbor algorithm
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);

        // Perform the search
        nanoflann::SearchParameters searchParams;
        searchParams.eps = approxEpsilon;
        searchParams.sorted = true;
        kdtree.findNeighbors(resultSet, query_pt, searchParams);

        // Check if correspondence is valid (within outlier threshold if set)
        if (outlierThreshold < 0 || out_dist_sqr <= outlierThreshold * outlierThreshold)
        {
            // Get the target point
            const Point3D &targetPoint = target.points[ret_index];

            // Accumulate sums for centroids
            sum_source += Eigen::Vector3d(sourcePoint.x, sourcePoint.y, sourcePoint.z);
            sum_target += Eigen::Vector3d(targetPoint.x, targetPoint.y, targetPoint.z);

            // Accumulate sums for cross-covariance elements
            double sx = sourcePoint.x;
            double sy = sourcePoint.y;
            double sz = sourcePoint.z;
            double tx = targetPoint.x;
            double ty = targetPoint.y;
            double tz = targetPoint.z;

            // Accumulate cross-covariance matrix elements
            partialSums[0] += sx * tx;
            partialSums[1] += sx * ty;
            partialSums[2] += sx * tz;
            partialSums[3] += sy * tx;
            partialSums[4] += sy * ty;
            partialSums[5] += sy * tz;
            partialSums[6] += sz * tx;
            partialSums[7] += sz * ty;
            partialSums[8] += sz * tz;

            // Store the sums of coordinates
            partialSums[9] += sx;
            partialSums[10] += sy;
            partialSums[11] += sz;
            partialSums[12] += tx;
            partialSums[13] += ty;
            partialSums[14] += tz;

            numCorrespondences++;
        }
    }

    // Store the number of correspondences in the last element
    partialSums[15] = static_cast<double>(numCorrespondences);

    return partialSums;
}

// Calculate transformation matrix from partial sums
Eigen::Matrix4d ParallelICP::calculateTransformationFromPartialSums(const std::vector<double> &partialSums, const std::vector<int> &numCorrespondences, int numProcesses)
{
    // Initialize sums for centroids and cross-covariance
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d sum_source(0, 0, 0);
    Eigen::Vector3d sum_target(0, 0, 0);
    int totalCorrespondences = 0;

    // Aggregate partial sums from all processes
    for (int p = 0; p < numProcesses; ++p)
    {
        const double *sums = &partialSums[p * 16];

        // Accumulate cross-covariance matrix
        H(0, 0) += sums[0];
        H(0, 1) += sums[1];
        H(0, 2) += sums[2];
        H(1, 0) += sums[3];
        H(1, 1) += sums[4];
        H(1, 2) += sums[5];
        H(2, 0) += sums[6];
        H(2, 1) += sums[7];
        H(2, 2) += sums[8];

        // Accumulate centroid components
        sum_source(0) += sums[9];
        sum_source(1) += sums[10];
        sum_source(2) += sums[11];
        sum_target(0) += sums[12];
        sum_target(1) += sums[13];
        sum_target(2) += sums[14];

        totalCorrespondences += numCorrespondences[p];
    }

    // If no valid correspondences found, return identity transform
    if (totalCorrespondences == 0)
    {
        return Eigen::Matrix4d::Identity();
    }

    // Compute centroids
    Eigen::Vector3d centroid_source = sum_source / totalCorrespondences;
    Eigen::Vector3d centroid_target = sum_target / totalCorrespondences;

    // Compute centered covariance matrix
    H -= (centroid_source * centroid_target.transpose()) * totalCorrespondences;

    // SVD decomposition to find rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure we have a rotation matrix (det=1)
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0)
    {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // Compute translation
    Eigen::Vector3d t = centroid_target - R * centroid_source;

    // Construct full 4x4 transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = R;
    transform.block<3, 1>(0, 3) = t;

    return transform;
}

// Calculate error metric
ParallelICP::PartialErrorResult ParallelICP::calculatePartialErrorSumWithKDTree(const PointCloud &localSource, const PointCloud &target, const KDTree &kdtree, double outlierThreshold, double approxEpsilon)
{
    PartialErrorResult result;
    result.sumSquaredError = 0.0;
    result.numCorrespondences = 0;

    // For each point in local source, find closest point in target using KD-tree
    for (const auto &sourcePoint : localSource.points)
    {
        // Setup query point
        double query_pt[3] = {sourcePoint.x, sourcePoint.y, sourcePoint.z};

        // Create result set for single nearest neighbor search
        size_t ret_index;
        double out_dist_sqr;

        // Find nearest neighbor with approx parameter
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);

        // Perform the search
        nanoflann::SearchParameters searchParams;
        searchParams.eps = approxEpsilon;
        searchParams.sorted = true;
        kdtree.findNeighbors(resultSet, query_pt, searchParams);

        // Check if correspondence is valid (within outlier threshold if set)
        if (outlierThreshold < 0 || out_dist_sqr <= outlierThreshold * outlierThreshold)
        {
            // Accumulate squared distance
            result.sumSquaredError += out_dist_sqr;
            result.numCorrespondences++;
        }
    }

    return result;
}

// Save colored point clouds
void ParallelICP::saveColoredPointClouds(const PointCloud &target, const PointCloud &source, const std::string &filename)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write target points (blue)
    for (const auto &p : target.points)
    {
        outFile << p.x << " " << p.y << " " << p.z << " 0 0 255" << std::endl;
    }

    // Write source points (red)
    for (const auto &p : source.points)
    {
        outFile << p.x << " " << p.y << " " << p.z << " 255 0 0" << std::endl;
    }

    outFile.close();
}
