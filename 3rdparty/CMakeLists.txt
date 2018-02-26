list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

include(benchmark.CMakeLists.txt)
include(boost.CMakeLists.txt)
include(opencl.CMakeLists.txt)
include(vtk.CMakeLists.txt)
include(pcl.CMakeLists.txt)
include(gtest.CMakeLists.txt)
