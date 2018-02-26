/**
 * @file Unit tests for normal computation
 */
#include <memory>
 
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>

#include "gtest.h"
#include "utils.h"

#include "normalComputation.h"
#include "knn.h"

using pcl::PointCloud;
using pcl::PointXYZ;

using kapernikov::test::generateCloud;
using kapernikov::test::generateIndices;

namespace  {
    const uint64_t MIN_RANGE = 2U << 13;
    const uint64_t MAX_RANGE = 2U << 20;
} // namespace 

namespace kapernikov {
    namespace test {
        SCENARIO(NormalComputationTest, comparisonTest, "Compare the CPU normal computation with the opencl version") {
            opencl::buildKernel();

            GIVEN("A knn cache") {
                for(unsigned int i = MIN_RANGE; i <= MAX_RANGE; i = i << 1U) {
                    const uint32_t TEST_SIZE = i;
                    auto cloud = generateCloud(TEST_SIZE);

                    std::vector<std::vector<int>> indices;
                    indices.reserve(cloud.size());
                    for(size_t i = 0U; i < cloud.size(); ++i) {
                        std::vector<int> new_indices = generateIndices(KNN_SIZE, TEST_SIZE-1U);
                        indices.emplace_back(new_indices);
                    }

                    Eigen::MatrixXi indices_as_matrix(indices.front().size(), indices.size());
                    for(size_t i = 0; i < indices.size(); ++i) {
                        for(size_t j = 0; j < indices.at(j).size(); ++j) {
                            indices_as_matrix(j, i) = indices.at(i).at(j);
                        }
                    }

                    ASSERT_GT(TEST_SIZE, KNN_SIZE);

                    WHEN("We compute the normals") {
                        auto expected_result = pcl::compute_normals(cloud, indices);
                        auto opencl_result = opencl::compute_normals(cloud, indices_as_matrix);

                        THEN("The size of the result should be bigger than 0") {
                            ASSERT_GT(expected_result->size(), 0U);
                            ASSERT_GT(opencl_result->size(), 0U);
                        }

                        THEN("The result must have the same dimensions") {
                            ASSERT_EQ(expected_result->size(), opencl_result->size());
                            ASSERT_EQ(expected_result->rows(), opencl_result->rows());
                            ASSERT_EQ(expected_result->cols(), opencl_result->cols());
                        }

                        THEN("The results should be equal") {
                            for(uint32_t row = 0U; row < expected_result->rows(); ++row) {
                                for(uint32_t col = 0U; col < expected_result->cols(); ++col) {
                                    //std::cout << "Row = " << row << " col = " << col << std::endl;
                                    ASSERT_NEAR((*expected_result)(row, col), (*opencl_result)(row, col), 14.0E-4);
                                }
                            }
                        }
                    }
                }
            }
        }
    } // namespace test
} // namespace kapernikov
