/**
 * @file    Utilities for generating (test) inputs
 */
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <random>

namespace kapernikov {
    namespace test {
        std::vector<int> generateIndices(unsigned int length, unsigned int max_index) noexcept {
            std::vector<int> indices;

            std::default_random_engine engine_int;
            std::uniform_int_distribution<int> dist_int(0,max_index);
            auto gen_int = std::bind(dist_int,engine_int);

            indices.reserve(length);
            while(indices.size() < length) {
                auto random_number = gen_int();

                // Check whether the random number is unique
                if(std::find(indices.begin(), indices.end(), random_number) == indices.end()) {
                    indices.emplace_back(random_number);
                }
            }
            return indices;
        }

        pcl::PointCloud<pcl::PointXYZ> generateCloud(unsigned int length) noexcept {
            pcl::PointCloud<pcl::PointXYZ> cloud;

            std::default_random_engine engine_float;
            std::uniform_real_distribution<float> dist_float;
            auto gen_float = std::bind(dist_float,engine_float);
  
            cloud.reserve(length);
            for(size_t i = 0; i < length; ++i) {
                cloud.push_back(pcl::PointXYZ(gen_float(), gen_float(), gen_float()));
            }
            return cloud;
        }

        void generatePointCloud(unsigned int length, pcl::PointCloud<pcl::PointXYZ>& cloud, std::vector<int>& indices) noexcept {
            indices = generateIndices(length, length);
            cloud = generateCloud(length);
        }
    } // namespace test
} // namespace kapernikov
