# parallel-icp

## single thread command to compile
g++ ICP.cpp ICP.h point3D.cpp point3D.h pointCloud.cpp pointCloud.h runner.cpp -I/home/divyanshu/eigen-3.4.0 -o icp_single_threa


## paraller thread command to compile
mpic++ parallel_ICP.cpp parallel_ICP.h point3D.cpp point3D.h pointCloud.h pointCloud.cpp parallel_runner.cpp -I/home/divyanshu/eigen-3.4.0 -o icp_mpi