/**
 * @file    Implementations of normal computation. The implementations are based on the PCL 1.8.1 implementation.
 */
#include "normalComputation.h"

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>

#include <boost/compute.hpp>
#include "knn.h"

namespace compute = boost::compute;

using std::vector;

using pcl::PointCloud;
using pcl::PointXYZ;

namespace {
compute::device device = compute::system::default_device();
compute::context context(device);
compute::command_queue queue(context, device);

const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    // NOTE: Define KNN_SIZE as a build parameter using "-D KNN_SIZE=<size>"
    __constant unsigned int KNN_BATCH_SIZE = KNN_SIZE;
    __constant float EPSILON = 1.19209e-07f;

    inline float4 computeRoots2(const float b, const float c) {
        //adapted from PCL source
        float x = (float)(0.0f);
        float d = b * b - 4.0f * c;
        d = select(
            d, 0.0f,
            d < 0.0f); // Use select instead of an if to prevent wavefront divergence.
                       // This condition should always evaluate to false: in the other case, this function is not used properly

        float sd = sqrt(d);

        float z = (float)(0.5f * (b + sd));
        float y = (float)(0.5f * (b - sd));
        return (float4)(x, y, z, 0); 
    }

    inline float4 computeRoots(const float4 row1, const float4 row2, const float4 row3) {
        //adapted from PCL source
        // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
        // eigenvalues are the roots to this equation, all guaranteed to be
        // real-valued, because the matrix is symmetric.
        float c0 = row1.x * row2.y * row3.z + 2.0f * row1.y * row1.z * row2.z -
                   row1.x * row2.z * row2.z - row2.y * row1.z * row1.z -
                   row3.z * row1.y * row1.y;

        float c1 = row1.x * row2.y - row1.y * row1.y + row1.x * row3.z -
                   row1.z * row1.z + row2.y * row3.z - row2.z * row2.z;

        float c2 = row1.x + row2.y + row3.z;


        float4 roots;
        if(fabs(c0) < EPSILON) { // one root is 0 -> quadratic equation
            roots = computeRoots2(c2, c1);
        }
        else {
            const float s_inv3 = (1.0f / 3.0f);
            const float s_sqrt3 = sqrt(3.0f);
            // Construct the parameters used in classifying the roots of the equation
            // and in solving the equation for the roots in closed form.
            float c2_over_3 = c2 * s_inv3;
            float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
            if(a_over_3 > 0.0f) {
                a_over_3 = 0.0f;
            }

            float half_b =
                0.5f * (c0 + c2_over_3 * (2.0f * c2_over_3 * c2_over_3 - c1));

            float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
            if(q > 0.0f) {
                q = 0.0f;
            }

            // Compute the eigenvalues by solving for the roots of the polynomial.
            float rho = sqrt(-a_over_3);
            float theta = atan2(sqrt(-q), half_b) * s_inv3;
            float cos_theta = cos(theta);
            float sin_theta = sin(theta);

            roots.x = c2_over_3 + 2.0f * rho * cos_theta;
            roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
            roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

            // Sort in increasing order.
            if(roots.x >= roots.y) {
                roots.xy = roots.yx;
            }
            if(roots.y >= roots.z) {
                roots.yz = roots.zy;
                if(roots.x >= roots.y)
                    roots.xy = roots.yx;
            }

            if(roots.x <=
               0.0f) // eigenval for symetric positive semi-definite matrix can not be negative! Set it to 0
                roots = computeRoots2(c2, c1);
        }
        return roots;
    }

    // Note: the returned eigenvalues are not scaled back to the actual scale
    inline float4 eigen33(const float4 row1, const float4 row2, const float4 row3) {
        // Calculate the scale by determing the largest value of all rows
        float4 abs1 = fabs(row1);
        float4 abs2 = fabs(row2);
        float4 abs3 = fabs(row3);
        float4 max = fmax(abs1, fmax(abs2, abs3));
        float scale = fmax(max.x, fmax(max.y, max.z));

        if(scale == FLT_MIN) {
            scale = 1.0f;
        }

        float4 scaled_row1 = row1/scale;
        float4 scaled_row2 = row2/scale;
        float4 scaled_row3 = row3/scale;
         
        float4 roots = computeRoots(scaled_row1, scaled_row2, scaled_row3);
        return roots;
    }

    inline float4 centroid(const float4* indexed_points) {
        float4 centroid = (float4)(0, 0, 0, 0);

        // NOTE: We are using a numerically more stable algorithm for calculating the centroid.
        //       See https://diego.assencio.com/?index=c34d06f4f4de2375658ed41f70177d59
        for(unsigned int i = 0; i < KNN_BATCH_SIZE; ++i) {
            float4 additional_point = indexed_points[i];
            centroid += (additional_point - centroid) / (i + 1);
        }
        return centroid;
    }

    inline void covariance(const float4* indexed_points, const float4 centroid, float4* cov_matrix_row1, float4* cov_matrix_row2, float4* cov_matrix_row3) {
        *cov_matrix_row1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        *cov_matrix_row2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        *cov_matrix_row3 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for(unsigned int i = 0; i < KNN_BATCH_SIZE; ++i) {
            float4 pt = indexed_points[i] - centroid;
            cov_matrix_row2->y += pt.y * pt.y;
            cov_matrix_row2->z += pt.y * pt.z;
            cov_matrix_row3->z += pt.z * pt.z;
            pt = pt * pt.x;
            cov_matrix_row1->x += pt.x;
            cov_matrix_row1->y += pt.y;
            cov_matrix_row1->z += pt.z;
        }

        // fill in the other half of the matrix
        cov_matrix_row3->y = cov_matrix_row2->z;
        cov_matrix_row3->x = cov_matrix_row1->z;
        cov_matrix_row2->x = cov_matrix_row1->y;

        // Normalize the covariance matrix
        *cov_matrix_row1 /= KNN_BATCH_SIZE;
        *cov_matrix_row2 /= KNN_BATCH_SIZE;
        *cov_matrix_row3 /= KNN_BATCH_SIZE;
    }

    inline float squaredNorm(const float4 input) {
        return input.x * input.x + input.y * input.y + input.z * input.z;
    }

    inline bool isMuchSmallerThan(const float x, const float z) {
        return x * x <= EPSILON * EPSILON * z * z;
    }

    inline float4 unitOrthogonal(const float4 vector) {
        // Take (-y, z, 0) and normalize it, unless x and y are both close to zero. Take (0, -z, y) and normalize instead
        if(isMuchSmallerThan(vector.x, vector.z) && isMuchSmallerThan(vector.y, vector.z)) {
            float size = rsqrt(vector.y*vector.y + vector.z * vector.z);
            return (float4)(0.0f, -vector.z, vector.y, 0.0f)/size;
        }
        float size = rsqrt(vector.x*vector.x + vector.y * vector.y);
        return (float4)(-vector.y, vector.x, 0.0f, 0.0f)/size;
    }

    inline void index(__global const float4* cloud, __global const int* indices, float4* indexed_points) {
        const unsigned int start_offset = get_global_id(0) * KNN_BATCH_SIZE;
        for(unsigned int i = 0; i < KNN_BATCH_SIZE; ++i) {
            indexed_points[i] = cloud[indices[start_offset + i]];
        }
    }

    inline float4 associated_eigenvector(const float4 scaled_row1, const float4 scaled_row2, const float4 scaled_row3, const float eigenvalue) {
        scaled_row1.x -= eigenvalue;
        scaled_row2.y -= eigenvalue;
        scaled_row3.z -= eigenvalue;

        float4 vec1 = cross(scaled_row1, scaled_row2);
        float4 vec2 = cross(scaled_row1, scaled_row3); 
        float4 vec3 = cross(scaled_row2, scaled_row3); 

        float len1 = squaredNorm(vec1);
        float len2 = squaredNorm(vec2);
        float len3 = squaredNorm(vec3);

        float4 largest_eigenvector = len1 >= len2 ? vec1 : vec2;
        float largest_length = len1 >= len2 ? len1 : len2;
        largest_eigenvector = largest_length >= len3 ? largest_eigenvector : vec3;
        largest_length = largest_length >= len3 ? largest_length : len3;

        return largest_eigenvector/sqrt(largest_length);
    }

    inline float4 normal_from_covariance(const float4 row1, const float4 row2, const float4 row3) {
        // Note: Calculating the normal is eigenvector associated with the lowest eigenvalue. However, if the original shape was planer, the lowest eigenvalue will be very low, causing an inaccurate normal.
        // Therefore, the normal is calculated from the (normalized) cross product of the eigenvectors associated with the two largest eigenvalues

        // Calculate the scale by determing the largest value of all rows
        float4 abs1 = fabs(row1);
        float4 abs2 = fabs(row2);
        float4 abs3 = fabs(row3);
        float4 max = fmax(abs1, fmax(abs2, abs3));
        float scale = fmax(max.x, fmax(max.y, max.z));

        if(scale == FLT_MIN) {
            scale = 1.0f;
        }

        float4 scaled_row1 = row1/scale;
        float4 scaled_row2 = row2/scale;
        float4 scaled_row3 = row3/scale;
         
        // Note: computeRoots will output the eigenvalues in ascending order
        float4 eigenvalues = computeRoots(scaled_row1, scaled_row2, scaled_row3);

        if((eigenvalues.z - eigenvalues.x) <= EPSILON) {
            // All three are equal => the eigenvectors are the identity matrix
            // Return the first eigenvector as the normal
            return (float4)(-1.0f, 0.0f, 0.0f, 0.0f);
        }

        if((eigenvalues.z - eigenvalues.y) <= EPSILON) {
            // second and third equal
            float4 eigenvector1 = associated_eigenvector(scaled_row1, scaled_row2, scaled_row3, eigenvalues.x);

            // In this case, it makes little sense to calculate the third eigenvector from the first eigenvalue, only to calculate the normal based on this result. So in this case, just return the first eigenvector
            return -eigenvector1;
        }

        float4 eigenvector2;
        float4 eigenvector3 = associated_eigenvector(scaled_row1, scaled_row2, scaled_row3, eigenvalues.z);
        if((eigenvalues.y - eigenvalues.x) <= EPSILON) {
            // first and second equal
            eigenvector2 = unitOrthogonal(eigenvector3);
        } else {
            eigenvector2 = associated_eigenvector(scaled_row1, scaled_row2, scaled_row3, eigenvalues.y);

            // In order to improve unicity of the solution, we use the convention of making sure that the first coordinate value
            // of the eigenvector is positive
            eigenvector2 = eigenvector2.x >= 0 ? eigenvector2 : -eigenvector2;
        }
        return cross(eigenvector3, eigenvector2);
    }

    __kernel void normals(__global const float4* points,
                              __global const int* indices,
                              __global float4* eigenvalues) {
        // Convert indices to an actual array of ordered points
        float4 indexed_points[KNN_SIZE];
        index(points, indices, indexed_points);

        // Calculate centroid
        float4 element_centroid = centroid(indexed_points);

        // Calculate the covariance of the neighbourhood
        float4 cov_matrix_row1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 cov_matrix_row2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 cov_matrix_row3 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        covariance(indexed_points, element_centroid, &cov_matrix_row1, &cov_matrix_row2, &cov_matrix_row3);
        
        // Calculate the eigen values based on the covariance matrix
        eigenvalues[get_global_id(0)] = normal_from_covariance(cov_matrix_row1, cov_matrix_row2, cov_matrix_row3);
    }
);

compute::program filter_program =
    compute::program::create_with_source(source, context);
} // namespace

namespace kapernikov {
    namespace pcl {
        SigmaMatrixPtr compute_normals(const PointCloud<PointXYZ>& cloud, const vector<vector<int>>& neighbour_indices) noexcept {
            assert(neighbour_indices.size() == cloud.size());

            SigmaMatrixPtr normals(new SigmaMatrix(cloud.size(), 3));
#pragma omp parallel for
            for(size_t idx = 0U; idx < cloud.size(); idx++) {
                const vector<int>& indices = neighbour_indices.at(idx);

                Eigen::Vector4f centroid;
                ::pcl::compute3DCentroid(cloud, indices, centroid);

                // warning, PCL is INACCURATE in computing the covariance matrix!
                // see https://github.com/PointCloudLibrary/pcl/issues/560
                // this is enough to completely break the algorithm.
                // even with the workaround below, the accuracy is not very good (compared to numpy) but good enough.
                EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix(3, 3);
                if(::pcl::computeCovarianceMatrixNormalized(cloud, indices,
                                                          centroid,
                                                          covariance_matrix) != 0) {
                    // now do PCA!
                    EIGEN_ALIGN16 Eigen::Vector3f eigen_value{0, 0, 0};
                    EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
                    ::pcl::eigen33(covariance_matrix, eigen_vectors, eigen_value);

                    assert(eigen_value(2) >= eigen_value(0));
                    assert(eigen_value(2) >= eigen_value(1));

                    auto eigenVector1 = eigen_vectors.col(2);
                    auto eigenVector2 = eigen_vectors.col(1);

                    // Apply the convention where the first coordinate of the second eigenvector is positive
                    if(eigenVector2(0) < 0) {
                        eigenVector2 = -eigenVector2;
                    }
                    auto normal = eigenVector1.cross(eigenVector2);
                    (*normals)(idx, 0) = normal(0);
                    (*normals)(idx, 1) = normal(1);
                    (*normals)(idx, 2) = normal(2);
                } else {
                    (*normals)(idx, 0) = std::numeric_limits<float>::quiet_NaN();
                    (*normals)(idx, 0) = std::numeric_limits<float>::quiet_NaN();
                    (*normals)(idx, 0) = std::numeric_limits<float>::quiet_NaN();
                }
            }
            return normals;
        }
    }

    namespace opencl {
        bool buildKernel() noexcept {
          try {
            filter_program.build("-D KNN_SIZE=" + std::to_string(KNN_SIZE));
            return true;
          } catch (compute::opencl_error e) {
            std::cout << "Build Error: " << std::endl << filter_program.build_log();
          }
          return false;
        }

        SigmaMatrixPtr compute_normals(const PointCloud<PointXYZ>& cloud, const Eigen::MatrixXi& neighbour_indices) noexcept {
            assert(static_cast<size_t>(neighbour_indices.cols()) == cloud.size());

            unsigned int total_points = neighbour_indices.cols();

            // Copy the indices to the gpu
            auto indices_bufsize = neighbour_indices.size() * sizeof(int);
            compute::buffer indices_device(context, indices_bufsize, CL_MEM_READ_ONLY);
            queue.enqueue_write_buffer(indices_device, 0, indices_bufsize, neighbour_indices.data());

            // Copy the cloud to the gpu
            auto bufsize = cloud.size() * sizeof(PointXYZ);
            compute::buffer cloud_device(context, bufsize, CL_MEM_READ_ONLY);
            queue.enqueue_write_buffer(cloud_device, 0, bufsize, cloud.points.data());

            // Allocate output
            compute::vector<compute::float4_> output_device(total_points, context);

            // create filter kernel and set arguments
            compute::kernel filter_kernel(filter_program, "normals");

            filter_kernel.set_arg(0, cloud_device);
            filter_kernel.set_arg(1, indices_device);
            filter_kernel.set_arg(2, output_device);

            // Do the openCL calculation
            queue.enqueue_1d_range_kernel(filter_kernel, 0, total_points, 0);

            // Copy back the output
            vector<compute::float4_> normal_collection(output_device.size(), compute::float4_(0.0f, 0.0f, 0.0f, 0.0f));
            compute::copy(output_device.begin(), output_device.end(), normal_collection.begin(), queue);

            SigmaMatrixPtr results(new SigmaMatrix(total_points, 3));
            for(unsigned int i = 0U; i < total_points; ++i) {
                const compute::float4_ normal = normal_collection[i];
                for(unsigned int j = 0U; j < 3U; ++j) {
                    (*results)(i, j) = normal[j];
                }
            }
            return results;
        }
    } // namespace opencl
} // namespace kapernikov
