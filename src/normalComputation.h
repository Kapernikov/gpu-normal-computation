#ifndef _NORMAL_COMPUTATION_INCLUDE
#define _NORMAL_COMPUTATION_INCLUDE

/**
 * @file    Declares the various implementations of normal computation.
 */

#include <memory>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace kapernikov {
    using SigmaMatrix = Eigen::MatrixXf;
    using SigmaMatrixPtr = std::shared_ptr<Eigen::MatrixXf>;

    /**
     * Functions in this namespace use the pcl implementation
     */
    namespace pcl {
        /**
         * Compute the normals of the given cloud. The normal calculation will be based on the given neighbourhood indices.
         *
         * @param[in] cloud                 The point cloud to be used
         * @param[in] neighbour_indices     For each point in the given cloud this parameter contains a collection of the indices of the neighbouring points on the same index
         * @returns A matrix denoting a vector of 3 dimensional normals. Each row in the matrix is the normal associated with the neighbours of the respective point in the given cloud.
         */
        SigmaMatrixPtr compute_normals(const ::pcl::PointCloud<::pcl::PointXYZ>& cloud, const std::vector<std::vector<int>>& neighbour_indices) noexcept;
    } // namespace pcl

    /**
     * Functions in this namespace use the openCL implementation
     */
    namespace opencl {
        /**
         * Builds and initializes the opencl kernel for the currently running system.
         */
        bool buildKernel() noexcept;

        /**
         * Compute the normals of the given cloud. The normal calculation will be based on the given neighbourhood indices.
         *
         * @pre The buildKernel() function must have been called in order to initialize the kernel. Initializing once suffices for multiple calls to this function.
         * @param[in] cloud                 The point cloud to be used
         * @param[in] neighbour_indices     A matrix where each column is associated with the point with the same index in the given cloud. Each column itself consists of the indices of the neighbouring points of the associated point.
         * @returns A matrix denoting a vector of 3 dimensional normals. Each row in the matrix is the normal associated with the neighbours of the respective point in the given cloud.
         */
        SigmaMatrixPtr compute_normals(const ::pcl::PointCloud<::pcl::PointXYZ>& cloud, const Eigen::MatrixXi& neighbour_indices) noexcept;
    } // namespace opencl
} // namespace kapernikov

#endif  /* _NORMAL_COMPUTATION_INCLUDE */
