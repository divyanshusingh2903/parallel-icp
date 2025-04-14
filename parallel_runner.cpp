#include "parallel_ICP.h"
#include "pointCloud.h"
#include "point3D.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <mpi.h>  // Include MPI header
#include <chrono> // For timing

// Function to load a point cloud from an XYZ file
PointCloud loadXYZFile(const std::string &filename)
{
    PointCloud cloud;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return cloud;
    }

    std::string line;
    double x, y, z;
    // Optional RGB values that might be in the file
    double r, g, b;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);

        // Try to read XYZ coordinates
        if (iss >> x >> y >> z)
        {
            cloud.addPoint(Point3D(x, y, z));
        }
    }

    return cloud;
}

void printUsage(const char *programName)
{
    std::cout << "Usage: " << programName << " [-s source.xyz] [-t target.xyz] [-o output_prefix] "
              << "[-i max_iterations] [-c convergence_threshold] [-r outlier_threshold] [-v save_interval]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -s  Source point cloud file (default: source.xyz)" << std::endl;
    std::cout << "  -t  Target point cloud file (default: target.xyz)" << std::endl;
    std::cout << "  -o  Output file prefix (default: picp_result)" << std::endl;
    std::cout << "  -i  Maximum number of iterations (default: 6)" << std::endl;
    std::cout << "  -c  Convergence threshold (default: 1e-6)" << std::endl;
    std::cout << "  -r  Outlier rejection threshold (default: 0.1)" << std::endl;
    std::cout << "  -v  Save interval for intermediate results (default: 2)" << std::endl;
    std::cout << "  -h  Display this help message" << std::endl;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default file names and parameters
    std::string sourceFile = "source.xyz";
    std::string targetFile = "target.xyz";
    std::string outputFile = "picp_result";
    int maxIterations = 5;
    double convergenceThreshold = 1e-6;
    double outlierThreshold = 0.1;
    int saveInterval = 1;

    // Only root process prints help if no arguments
    if (rank == 0 && argc == 1)
    {
        printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-s" && i + 1 < argc)
        {
            sourceFile = argv[i + 1];
            i++;
        }
        else if (arg == "-t" && i + 1 < argc)
        {
            targetFile = argv[i + 1];
            i++;
        }
        else if (arg == "-o" && i + 1 < argc)
        {
            outputFile = argv[i + 1];
            i++;
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            maxIterations = std::stoi(argv[i + 1]);
            i++;
        }
        else if (arg == "-c" && i + 1 < argc)
        {
            convergenceThreshold = std::stod(argv[i + 1]);
            i++;
        }
        else if (arg == "-r" && i + 1 < argc)
        {
            outlierThreshold = std::stod(argv[i + 1]);
            i++;
        }
        else if (arg == "-v" && i + 1 < argc)
        {
            saveInterval = std::stoi(argv[i + 1]);
            i++;
        }
        else if (arg == "-h" || arg == "--help")
        {
            if (rank == 0)
            {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0)
    {
        std::cout << "\nRunning Parallel ICP with " << size << " processes" << std::endl;
        std::cout << "ICP Parameters:" << std::endl;
        std::cout << "  Source file: " << sourceFile << std::endl;
        std::cout << "  Target file: " << targetFile << std::endl;
        std::cout << "  Output prefix: " << outputFile << std::endl;
        std::cout << "  Max iterations: " << maxIterations << std::endl;
        std::cout << "  Convergence threshold: " << convergenceThreshold << std::endl;
        std::cout << "  Outlier rejection threshold: " << outlierThreshold << std::endl;
        std::cout << "  Save interval: " << saveInterval << std::endl;
        std::cout << std::endl;
    }

    // Ensure all processes use the same parameters
    MPI_Bcast(sourceFile.data(), sourceFile.length(), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(targetFile.data(), targetFile.length(), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(outputFile.data(), outputFile.length(), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxIterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&convergenceThreshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&outlierThreshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&saveInterval, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All processes load point clouds
    PointCloud source = loadXYZFile(sourceFile);
    PointCloud target = loadXYZFile(targetFile);

    if (rank == 0)
    {
        std::cout << "Loaded " << source.size() << " points from " << sourceFile << std::endl;
        std::cout << "Loaded " << target.size() << " points from " << targetFile << std::endl;
    }

    if (source.size() == 0 || target.size() == 0)
    {
        if (rank == 0)
        {
            std::cerr << "Failed to load point clouds. Exiting." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Set up parallel ICP parameters
    ParallelICP::Parameters params;
    params.maxIterations = maxIterations;
    params.convergenceThreshold = convergenceThreshold;
    params.outlierRejectionThreshold = outlierThreshold;
    params.outputFilePrefix = outputFile;
    params.saveInterval = saveInterval;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run parallel ICP algorithm
    Eigen::Matrix4d finalTransform = ParallelICP::align(source, target, params);

    // Stop timer
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Only the root process should report results
    if (rank == 0)
    {
        std::cout << "\nFinal transformation matrix:" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << finalTransform << std::endl;

        std::cout << "\nPoint cloud files saved:" << std::endl;
        std::cout << "- Initial state: " << outputFile << "_initial.xyzrgb" << std::endl;
        std::cout << "- Iteration files: " << outputFile << "_iter*.xyzrgb" << std::endl;
        std::cout << "- Final aligned result: " << outputFile << "_final.xyzrgb" << std::endl;
        std::cout << "- Aligned source only (green): " << outputFile << "_aligned.xyz" << std::endl;
        std::cout << "\nTarget is blue, source is red, aligned source is green." << std::endl;

        // Display timing information
        std::cout << "\nExecution time: " << elapsedSeconds.count() << " seconds" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}