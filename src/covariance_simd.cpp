/**
 * @file File containing a vectorized version of the computeCovarianceMatrix() function of pcl for dense point clouds. Interface and implementation based on \see http://docs.pointclouds.org/1.8.1/centroid_8hpp_source.html#l00263
 */
#include <boost/compute.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace compute = boost::compute;

namespace {
compute::device device = compute::system::default_device();
compute::context context(device);
compute::command_queue queue(context, device);
} // namespace

unsigned int
computeCovarianceMatrix(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                        const std::vector<int>& indices,
                        const Eigen::Vector4f& centroid,
                        Eigen::Matrix3f& covariances) noexcept {
    if(!cloud.is_dense) {
        std::cout << "Currently only dense point clouds are supported"
                  << std::endl;
        return 0;
    }

    std::vector<float> host_x;
    host_x.reserve(indices.size());
    std::vector<float> host_y;
    host_y.reserve(indices.size());
    std::vector<float> host_z;
    host_z.reserve(indices.size());

    for(auto index : indices) {
        host_x.emplace_back(cloud[index].x);
        host_y.emplace_back(cloud[index].y);
        host_z.emplace_back(cloud[index].z);
    }

    // create a vector of each dimension on the device
    compute::vector<float> x(host_x.begin(), host_x.end(), queue);
    compute::vector<float> y(host_y.begin(), host_y.end(), queue);
    compute::vector<float> z(host_z.begin(), host_z.end(), queue);

    // Create the centroid scalar as a vector repeating the scalar
    compute::vector<float> centroid_x(host_x.size(), centroid[0], queue);
    compute::vector<float> centroid_y(host_y.size(), centroid[1], queue);
    compute::vector<float> centroid_z(host_z.size(), centroid[2], queue);

    // Create the intermediate pt
    compute::vector<float> pt_x(x.size(), context);
    compute::vector<float> pt_y(y.size(), context);
    compute::vector<float> pt_z(z.size(), context);

    // Create pt
    compute::transform(x.begin(), x.end(), centroid_x.begin(), pt_x.begin(),
                       compute::minus<float>(), queue);
    compute::transform(y.begin(), y.end(), centroid_y.begin(), pt_y.begin(),
                       compute::minus<float>(), queue);
    compute::transform(z.begin(), z.end(), centroid_z.begin(), pt_z.begin(),
                       compute::minus<float>(), queue);

    // Calculate the covariance values
    float covariance_xx;
    compute::transform_reduce(pt_x.begin(), pt_x.end(), pt_x.begin(),
                              &covariance_xx, compute::multiplies<float>(),
                              compute::plus<float>(), queue);

    float covariance_xy;
    compute::transform_reduce(pt_x.begin(), pt_x.end(), pt_y.begin(),
                              &covariance_xy, compute::multiplies<float>(),
                              compute::plus<float>(), queue);

    float covariance_xz;
    compute::transform_reduce(pt_x.begin(), pt_x.end(), pt_z.begin(),
                              &covariance_xz, compute::multiplies<float>(),
                              compute::plus<float>(), queue);

    float covariance_yy;
    compute::transform_reduce(pt_y.begin(), pt_y.end(), pt_y.begin(),
                              &covariance_yy, compute::multiplies<float>(),
                              compute::plus<float>(), queue);

    float covariance_yz;
    compute::transform_reduce(pt_y.begin(), pt_y.end(), pt_z.begin(),
                              &covariance_yz, compute::multiplies<float>(),
                              compute::plus<float>(), queue);

    float covariance_zz;
    compute::transform_reduce(pt_z.begin(), pt_z.end(), pt_z.begin(),
                              &covariance_zz, compute::multiplies<float>(),
                              compute::plus<float>(), queue);

    covariances(0, 0) = covariance_xx;
    covariances(0, 1) = covariance_xy;
    covariances(0, 2) = covariance_xz;
    covariances(1, 1) = covariance_yy;
    covariances(1, 2) = covariance_yz;
    covariances(2, 2) = covariance_zz;

    // The covariance matrix is symmetric: mirror the elements over the diagonal
    covariances(1, 0) = covariances(0, 1);
    covariances(2, 0) = covariances(0, 2);
    covariances(2, 1) = covariances(1, 2);
    return indices.size();
}
