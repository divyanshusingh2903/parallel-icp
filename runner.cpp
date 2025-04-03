#include "ICP.h"
#include "pointCloud.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

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

    std::cout << "Loaded " << cloud.size() << " points from " << filename << std::endl;
    return cloud;
}

int main(int argc, char *argv[])
{
    // Default file names
    std::string sourceFile = "source.xyz";
    std::string targetFile = "target.xyz";
    std::string outputFile = "icp_result";

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
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [-s source.xyz] [-t target.xyz] [-o output_prefix]" << std::endl;
            return 0;
        }
    }

    // Load source and target point clouds
    PointCloud source = loadXYZFile(sourceFile);
    PointCloud target = loadXYZFile(targetFile);

    if (source.size() == 0 || target.size() == 0)
    {
        std::cerr << "Failed to load point clouds. Exiting." << std::endl;
        return 1;
    }

    // Set up ICP parameters
    ICP::Parameters params;
    params.maxIterations = 10;
    params.convergenceThreshold = 1e-6;
    params.outlierRejectionThreshold = 0.1; // Reject outliers further than 0.1 units
    params.outputFilePrefix = outputFile;   // Set output file prefix for ICP
    params.saveInterval = 5;                // Save intermediate files every 5 iterations

    // Run ICP algorithm - it will generate intermediate files internally
    Eigen::Matrix4d finalTransform = ICP::align(source, target, params);

    std::cout << "Final transformation matrix:" << std::endl;
    std::cout << finalTransform << std::endl;

    std::cout << "\nPoint cloud files saved:" << std::endl;
    std::cout << "- Initial state: " << outputFile << "_initial.xyz" << std::endl;
    std::cout << "- Iteration files: " << outputFile << "_iter*.xyz" << std::endl;
    std::cout << "- Final aligned result: " << outputFile << "_final.xyz" << std::endl;
    std::cout << "- Aligned source only (green): " << outputFile << "_aligned.xyz" << std::endl;
    std::cout << "\nTarget is blue, source is red, aligned source is green." << std::endl;

    return 0;
}