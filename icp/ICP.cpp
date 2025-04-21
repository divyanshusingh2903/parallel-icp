#include "ICP.h"

Eigen::Matrix4d ICP::align(const PointCloud &source, const PointCloud &target, const Parameters &params)
{
    // Initialize transformation matrix as identity
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();

    // Make a copy of the source point cloud that we'll transform
    PointCloud transformedSource = source;

    // Save initial state if output prefix is provided
    if (!params.outputFilePrefix.empty())
    {
        saveColoredPointClouds(target, transformedSource, params.outputFilePrefix + "_initial.xyzrgb");
    }

    double prevError = std::numeric_limits<double>::max();

    // Main ICP loop
    for (int iter = 0; iter < params.maxIterations; ++iter)
    {
        // 1. Find correspondences
        std::vector<std::pair<Point3D, Point3D>> correspondences;
        double currentError = findCorrespondences(transformedSource, target, correspondences, params.outlierRejectionThreshold);

        // Check if we have enough correspondences
        if (correspondences.size() < 3)
        {
            std::cerr << "Too few correspondences found: " << correspondences.size() << std::endl;
            break;
        }

        // 2. Calculate optimal transformation
        Eigen::Matrix4d currentTransform = calculateTransformation(correspondences);

        // 3. Update the accumulated transformation
        transformation = currentTransform * transformation;

        // 4. Apply the current transformation to the source point cloud
        transformedSource = source;                  // Reset to original
        transformedSource.transform(transformation); // Apply accumulated transformation

        // 5. Save intermediate results if requested
        if (!params.outputFilePrefix.empty() &&
            (iter % params.saveInterval == 0 || iter == params.maxIterations - 1))
        {
            saveColoredPointClouds(target, transformedSource,
                                   params.outputFilePrefix + "_iter" + std::to_string(iter + 1) + ".xyzrgb");
        }

        // 6. Check for convergence
        double improvement = prevError - currentError;
        prevError = currentError;

        std::cout << "Iteration " << iter + 1 << ", Error: " << currentError
                  << ", Improvement: " << improvement << std::endl;

        if (std::abs(improvement) < params.convergenceThreshold)
        {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }
    }

    // Save final state if output prefix is provided
    if (!params.outputFilePrefix.empty())
    {
        PointCloud finalSource = source;
        finalSource.transform(transformation);
        saveColoredPointClouds(target, finalSource, params.outputFilePrefix + "_final.xyzrgb");

        // Save aligned source separately (green color)
        std::ofstream alignedFile(params.outputFilePrefix + "_aligned.xyzrgb");
        alignedFile << std::fixed << std::setprecision(6);
        for (const auto &p : finalSource.points)
        {
            alignedFile << p.x << " " << p.y << " " << p.z << " 66 245 66" << std::endl; // Green
        }
        alignedFile.close();
    }

    return transformation;
}

double ICP::findCorrespondences(const PointCloud &source, const PointCloud &target, std::vector<std::pair<Point3D, Point3D>> &correspondences, double outlierThreshold)
{
    correspondences.clear();
    double totalError = 0.0;

    // For each point in source, find closest point in target
    for (const auto &sourcePoint : source.points)
    {
        double minDist = std::numeric_limits<double>::max();
        Point3D closestPoint;

        // Find closest point in target
        for (const auto &targetPoint : target.points)
        {
            double dist = sourcePoint.distance(targetPoint);
            if (dist < minDist)
            {
                minDist = dist;
                closestPoint = targetPoint;
            }
        }

        // Reject outliers if threshold is provided
        if (outlierThreshold < 0 || minDist < outlierThreshold)
        {
            correspondences.push_back(std::make_pair(sourcePoint, closestPoint));
            totalError += minDist * minDist;
        }
    }

    // Calculate mean squared error
    return correspondences.empty() ? std::numeric_limits<double>::max()
                                   : totalError / correspondences.size();
}

Eigen::Matrix4d ICP::calculateTransformation(const std::vector<std::pair<Point3D, Point3D>> &correspondences)
{
    // Compute centroids
    Eigen::Vector3d sourceCentroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetCentroid = Eigen::Vector3d::Zero();

    for (const auto &corr : correspondences)
    {
        sourceCentroid += Eigen::Vector3d(corr.first.x, corr.first.y, corr.first.z);
        targetCentroid += Eigen::Vector3d(corr.second.x, corr.second.y, corr.second.z);
    }

    sourceCentroid /= correspondences.size();
    targetCentroid /= correspondences.size();

    // Compute covariance matrix
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

    for (const auto &corr : correspondences)
    {
        Eigen::Vector3d sourcePoint(corr.first.x, corr.first.y, corr.first.z);
        Eigen::Vector3d targetPoint(corr.second.x, corr.second.y, corr.second.z);

        Eigen::Vector3d sourcePointCentered = sourcePoint - sourceCentroid;
        Eigen::Vector3d targetPointCentered = targetPoint - targetCentroid;

        covariance += targetPointCentered * sourcePointCentered.transpose();
    }

    // Compute optimal rotation using SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();

    // Ensure we have a proper rotation matrix (det = 1)
    if (rotation.determinant() < 0)
    {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) = -V.col(2);
        rotation = svd.matrixU() * V.transpose();
    }

    // Compute translation
    Eigen::Vector3d translation = targetCentroid - rotation * sourceCentroid;

    // Build transformation matrix
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    return transformation;
}

void ICP::saveColoredPointClouds(const PointCloud &target, const PointCloud &source, const std::string &filename)
{
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(6);

    // Write target points (blue)
    for (const auto &p : target.points)
    {
        file << p.x << " " << p.y << " " << p.z << " 66 132 245" << std::endl; // Blue
    }

    // Write source points (red)
    for (const auto &p : source.points)
    {
        file << p.x << " " << p.y << " " << p.z << " 245 66 66" << std::endl; // Red
    }

    file.close();
}