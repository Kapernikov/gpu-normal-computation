/**
 * @file    Benchmarks for the normal computation implementations
 */
#include <chrono>

#include <benchmark/benchmark.h>

#include "normalComputation.h"
#include "knn.h"
#include "utils.h"

using kapernikov::test::generateCloud;
using kapernikov::test::generateIndices;
using kapernikov::KNN_SIZE;

namespace  {
    const uint64_t MIN_RANGE = 2U << 13;
    const uint64_t MAX_RANGE = 2U << 20;
} // namespace 

static void normal_computation_cpu(benchmark::State& state) {
    for (auto _ : state) {
        auto cloud = generateCloud(state.range(0));
        std::vector<std::vector<int>> indices;
        indices.reserve(cloud.size());
        for(size_t i = 0U; i < cloud.size(); ++i) {
            std::vector<int> new_indices = generateIndices(KNN_SIZE, state.range(0)-1U);
            indices.emplace_back(new_indices);
        }

        // In order to be able to do an honest benchmark measurement, this benchmark is measured
        // in the same way as the GPU benchmark(s) are measured.
        auto start = std::chrono::high_resolution_clock::now();
        auto result = kapernikov::pcl::compute_normals(cloud, indices);
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(normal_computation_cpu)->RangeMultiplier(2)->Range(MIN_RANGE, MAX_RANGE)->UseManualTime()->Unit(benchmark::kMillisecond);


static void normal_computation_opencl(benchmark::State& state) {
    kapernikov::opencl::buildKernel();

    for (auto _ : state) {
        auto cloud = generateCloud(state.range(0));
        Eigen::MatrixXi indices(KNN_SIZE, cloud.size());
        for(size_t i = 0U; i < cloud.size(); ++i) {
            std::vector<int> new_indices = generateIndices(KNN_SIZE, state.range(0)-1U);
            for(size_t j = 0U; j < new_indices.size(); ++j) {
                indices(j, i) = new_indices.at(j);
            }
        }

        // Since this is a GPU calculation, do a manual timing
        auto start = std::chrono::high_resolution_clock::now();
        auto result = kapernikov::opencl::compute_normals(cloud, indices);
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(normal_computation_opencl)->RangeMultiplier(2)->Range(MIN_RANGE, MAX_RANGE)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
