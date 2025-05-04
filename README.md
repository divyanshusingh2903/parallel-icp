# Parallel ICP Algorithm for Point Cloud Registration

This repository contains implementations of the Iterative Closest Point (ICP) algorithm for 3D point cloud registration with several parallelization strategies using MPI.

## Overview

Point cloud registration is fundamental in 3D data processing applications like robotics, medical imaging, and autonomous navigation. This project accelerates the Iterative Closest Point (ICP) algorithm through parallelization using MPI and further enhances performance through dynamic resolution adjustment and spatial indexing via KD-trees.

## Implementations

This repository includes three main implementations:

1. **Basic Parallel ICP** - Distributes the workload across multiple MPI processes
2. **Parallel ICP with Dynamic Resolution** - Uses multi-resolution approach for faster convergence
3. **Parallel ICP with Spatial Structures** - Incorporates KD-trees for efficient nearest neighbor search

## Directory Structure

```
.
├── README.md
├── data                          # Contains source and target point clouds
├── executables                   # Compiled binaries
│   ├── icp_mpi
│   ├── icp_mpi_dynamic_resolution
│   └── icp_single_thread
├── icp                           # Single-threaded ICP implementation
├── parallel-icp                  # Basic parallel ICP implementation
├── parallel-icp-dynamic-resolution # Parallel ICP with multi-resolution
├── parallel-icp-spatial-structure  # Parallel ICP with KD-trees
├── point-cloud                   # Core point cloud data structures
├── output                        # Results from various experiments
│   ├── armadillo-multi-thread
│   ├── armadillo-multi-thread-dynamic-resolution
│   ├── armadillo-multi-thread-spatial-structure
│   ├── armadillo-single-thread
│   ├── strong-scaling            # Strong scaling test results
│   └── weak-scaling              # Weak scaling test results
└── xyzrgb_to_ply.py              # Conversion utility
```

## Algorithm Details

### 1. Parallel ICP

The basic parallel ICP algorithm:

1. Distributes the source point cloud among all available MPI processes
2. Each process independently computes:
   - Nearest neighbors for its subset of source points
   - Local centroids of corresponding pairs
   - Local covariance matrices
3. Gathers partial results to the root process
4. Root process computes optimal rotation (R) and translation (T) using SVD
5. Broadcasts the computed transformation to all processes
6. Each process applies the transformation to its local points
7. Repeats until convergence

### 2. Parallel ICP with Dynamic Resolution

This variant refines registration from coarse to fine resolutions:

1. Predefines multiple downsampling levels
2. Starts with highly downsampled source and target clouds
3. At each level:
   - Runs the Parallel ICP algorithm until convergence
   - Uses looser convergence thresholds at coarse levels
4. Progressively moves to finer resolutions
5. Tightens convergence criteria at each finer level

### 3. Parallel ICP with Spatial Structures (KD-Trees)

This enhancement accelerates the nearest neighbor search:

1. Constructs a KD-Tree from the target point cloud
2. Broadcasts the KD-Tree to all processes
3. During correspondence search, each process queries the KD-Tree
4. Proceeds with the standard Parallel ICP algorithm

## Performance Results

### Weak Scaling

As the problem size increases with the number of processors, ideally the runtime should remain constant.

| # of Processes | Total Points | pICP (s) | pICP + Dynamic Resolution (s) | pICP + Spatial Structures (s) |
|----------------|--------------|----------|------------------------------|------------------------------|
| 2              | 20000        | 232.992  | 53.882                       | 13.433                       |
| 4              | 40000        | 375.279  | 104.268                      | 18.962                       |
| 8              | 80000        | 921.719  | 278.796                      | 51.683                       |
| 16             | 160000       | 2638.773 | 778.511                      | 138.429                      |

### Strong Scaling

As the number of processors increases for a fixed problem size, the runtime should decrease.

| # of Processes | Total Points | pICP (s) | pICP + Dynamic Resolution (s) | pICP + Spatial Structures (s) |
|----------------|--------------|----------|------------------------------|------------------------------|
| 2              | 80000        | 2479.270 | 784.718                      | 32.729                       |
| 4              | 80000        | 1845.714 | 455.621                      | 38.630                       |
| 8              | 80000        | 921.719  | 278.796                      | 51.683                       |
| 16             | 80000        | 762.797  | 234.630                      | 76.936                       |

### Key Findings

- The spatial structures approach provides dramatic performance improvements, with up to 76x speedup over the base implementation
- Dynamic resolution offers a balanced approach with good absolute performance and reasonable scaling behavior
- Algorithm selection should be guided by available hardware resources and problem characteristics

## Getting Started

### Prerequisites

- MPI implementation (e.g., OpenMPI, MPICH)
- C++ compiler with C++11 support
- CMake (optional, for building)

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/parallel-icp.git
cd parallel-icp

# Build the executables
make
```

### Running the Examples

```bash
# Run single-threaded version
./executables/icp_single_thread data/bunny-40000-source.xyz data/bunny-40000-target.xyz

# Run parallel version with 4 processes
mpirun -np 4 ./executables/icp_mpi data/bunny-40000-source.xyz data/bunny-40000-target.xyz

# Run parallel version with dynamic resolution and 8 processes
mpirun -np 8 ./executables/icp_mpi_dynamic_resolution data/bunny-40000-source.xyz data/bunny-40000-target.xyz
```

### Converting Output to PLY Format

```bash
python xyzrgb_to_ply.py input_file.xyzrgb output_file.ply
```

## References

- C. Langis, M. Greenspan and G. Godin, "The parallel iterative closest point algorithm," Proceedings Third International Conference on 3-D Digital Imaging and Modeling, 2001.
- Jerome H. Friedman, et al., "An Algorithm for Finding Best Matches in Logarithmic Expected Time," ACM Transactions on Mathematical Software, 1977.
- simpleICP implementations, GitHub repository: https://github.com/pglira/simpleICP

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Divyanshu Singh - CSCE-626 Final Project - April 28, 2025
