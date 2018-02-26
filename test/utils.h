#ifndef UTILS_INCLUDE
#define UTILS_INCLUDE

/**
 * @file    Utilities for generating (test) inputs
 */

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace kapernikov {
    namespace test {
        /**
         * Generate a collection of random unique indices.
         *
         * @param[in] length    The desired length of the random indices collection
         * @param[in] max_index The maximum index value to use in the generated collection
         * @returns A collection of unique random indices
         */
        std::vector<int> generateIndices(unsigned int length, unsigned int max_index) noexcept;

        /**
         * Generate a cloud of random points
         *
         * @param[in] length    The desired number of points in the resulting cloud
         * @returns A cloud of the given length
         */
        ::pcl::PointCloud<::pcl::PointXYZ> generateCloud(unsigned int length) noexcept;

        /**
         * Generate a point cloud with random points and associated indices
         *
         * @param[in] length    The desired number of points in the point cloud. The number of unique indices will be equal to the number of points in the cloud
         * @param[out] cloud    The generated point cloud
         * @param[out] indices  The generated indices associated with the generated point cloud
         */
        void generatePointCloud(unsigned int length, ::pcl::PointCloud<::pcl::PointXYZ>& cloud, std::vector<int>& indices) noexcept;
    } // namespace test
} // namespace kapernikov

#endif  /* UTILS_INCLUDE */
