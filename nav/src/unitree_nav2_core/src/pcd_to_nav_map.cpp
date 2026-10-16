#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <octomap/OcTree.h>
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace fs = std::filesystem;

namespace
{
struct Options
{
  fs::path input;
  fs::path output_dir;
  double resolution = 0.03;
  double padding = 0.50;
  double obstacle_min_height = 0.15;
  double obstacle_max_height = 1.80;
  double ground_band = 0.15;
  double obstacle_radius = 0.04;
  double free_radius = 0.15;
  bool ground_z_set = false;
  double ground_z = 0.0;
};

double asDouble(const std::string &value, const std::string &name)
{
  try
  {
    return std::stod(value);
  }
  catch (const std::exception &)
  {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

Options parseArguments(int argc, char **argv)
{
  Options options;
  std::unordered_map<std::string, std::string *> strings = {
    {"--input", nullptr}, {"--output-dir", nullptr}};
  for (int index = 1; index < argc; ++index)
  {
    const std::string arg = argv[index];
    if (arg == "--help")
    {
      std::cout
        << "Usage: pcd_to_nav_map --input map.pcd --output-dir DIR "
        << "[--resolution 0.03] [--ground-z auto|Z] "
        << "[--obstacle-min-height 0.15] [--obstacle-max-height 1.80] "
        << "[--ground-band 0.15] [--padding 0.50] "
        << "[--obstacle-radius 0.04] [--free-radius 0.15]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) throw std::runtime_error("Missing value after " + arg);
    const std::string value = argv[++index];
    if (arg == "--input") options.input = value;
    else if (arg == "--output-dir") options.output_dir = value;
    else if (arg == "--resolution") options.resolution = asDouble(value, arg);
    else if (arg == "--padding") options.padding = asDouble(value, arg);
    else if (arg == "--obstacle-min-height")
      options.obstacle_min_height = asDouble(value, arg);
    else if (arg == "--obstacle-max-height")
      options.obstacle_max_height = asDouble(value, arg);
    else if (arg == "--ground-band") options.ground_band = asDouble(value, arg);
    else if (arg == "--obstacle-radius") options.obstacle_radius = asDouble(value, arg);
    else if (arg == "--free-radius") options.free_radius = asDouble(value, arg);
    else if (arg == "--ground-z")
    {
      if (value != "auto")
      {
        options.ground_z_set = true;
        options.ground_z = asDouble(value, arg);
      }
    }
    else throw std::runtime_error("Unknown argument: " + arg);
  }
  if (options.input.empty() || options.output_dir.empty())
    throw std::runtime_error("--input and --output-dir are required");
  if (!(options.resolution > 0.0) || options.obstacle_min_height < 0.0 ||
      options.obstacle_max_height <= options.obstacle_min_height)
    throw std::runtime_error("Invalid resolution or obstacle height interval");
  return options;
}

double percentile(std::vector<float> values, double fraction)
{
  if (values.empty()) throw std::runtime_error("Point cloud contains no finite Z values");
  const std::size_t index = std::min(
    values.size() - 1,
    static_cast<std::size_t>(std::floor(fraction * static_cast<double>(values.size() - 1))));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

Eigen::Vector4f estimateGroundPlane(
  const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud, const Options &options)
{
  if (options.ground_z_set)
    return Eigen::Vector4f(0.0F, 0.0F, 1.0F, static_cast<float>(-options.ground_z));

  std::vector<float> z_values;
  z_values.reserve(cloud->size());
  for (const auto &point : cloud->points)
    if (pcl::isFinite(point)) z_values.push_back(point.z);
  const double low = percentile(z_values, 0.005) - 0.10;
  const double high = percentile(z_values, 0.60);

  pcl::PointCloud<pcl::PointXYZI>::Ptr candidates(
    new pcl::PointCloud<pcl::PointXYZI>());
  candidates->reserve(cloud->size());
  for (const auto &point : cloud->points)
    if (pcl::isFinite(point) && point.z >= low && point.z <= high)
      candidates->push_back(point);

  pcl::SACSegmentation<pcl::PointXYZI> segmentation;
  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setMethodType(pcl::SAC_RANSAC);
  segmentation.setAxis(Eigen::Vector3f::UnitZ());
  segmentation.setEpsAngle(0.30);
  segmentation.setDistanceThreshold(0.08);
  segmentation.setMaxIterations(1500);
  segmentation.setInputCloud(candidates);
  pcl::PointIndices inliers;
  pcl::ModelCoefficients coefficients;
  segmentation.segment(inliers, coefficients);

  if (coefficients.values.size() != 4 || inliers.indices.size() < 100)
  {
    const double histogram_floor = percentile(z_values, 0.25);
    return Eigen::Vector4f(0.0F, 0.0F, 1.0F, static_cast<float>(-histogram_floor));
  }
  Eigen::Vector4f plane(
    coefficients.values[0], coefficients.values[1],
    coefficients.values[2], coefficients.values[3]);
  const float norm = plane.head<3>().norm();
  plane /= norm;
  if (plane.z() < 0.0F) plane = -plane;
  return plane;
}

double heightAbovePlane(const pcl::PointXYZI &point, const Eigen::Vector4f &plane)
{
  return plane.x() * point.x + plane.y() * point.y + plane.z() * point.z + plane.w();
}

std::vector<std::uint8_t> dilate(
  const std::vector<std::uint8_t> &source, std::size_t width, std::size_t height,
  int radius_cells)
{
  if (radius_cells <= 0) return source;
  std::vector<std::uint8_t> output(source.size(), 0);
  for (std::size_t y = 0; y < height; ++y)
  {
    for (std::size_t x = 0; x < width; ++x)
    {
      if (!source[x + y * width]) continue;
      for (int dy = -radius_cells; dy <= radius_cells; ++dy)
      {
        for (int dx = -radius_cells; dx <= radius_cells; ++dx)
        {
          if (dx * dx + dy * dy > radius_cells * radius_cells) continue;
          const int nx = static_cast<int>(x) + dx;
          const int ny = static_cast<int>(y) + dy;
          if (nx >= 0 && ny >= 0 && nx < static_cast<int>(width) &&
              ny < static_cast<int>(height))
            output[static_cast<std::size_t>(nx) + static_cast<std::size_t>(ny) * width] = 1;
        }
      }
    }
  }
  return output;
}

void writePgm(
  const fs::path &path, const std::vector<std::int8_t> &grid,
  std::size_t width, std::size_t height)
{
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot write " + path.string());
  output << "P5\n# Unitree G1 PCL + OctoMap projection\n" << width << ' ' << height << "\n255\n";
  for (std::size_t image_y = 0; image_y < height; ++image_y)
  {
    const std::size_t grid_y = height - 1 - image_y;
    for (std::size_t x = 0; x < width; ++x)
    {
      const std::int8_t value = grid[x + grid_y * width];
      const unsigned char pixel = value == 100 ? 0 : (value == 0 ? 254 : 205);
      output.write(reinterpret_cast<const char *>(&pixel), 1);
    }
  }
}
}  // namespace

int main(int argc, char **argv)
{
  try
  {
    const Options options = parseArguments(argc, argv);
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw(
      new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile(options.input.string(), *raw) != 0)
      throw std::runtime_error("Cannot load PCD: " + options.input.string());

    pcl::PointCloud<pcl::PointXYZI>::Ptr finite(
      new pcl::PointCloud<pcl::PointXYZI>());
    finite->reserve(raw->size());
    for (const auto &point : raw->points)
      if (pcl::isFinite(point)) finite->push_back(point);
    if (finite->empty()) throw std::runtime_error("PCD has no finite XYZ points");

    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(finite);
    sor.setMeanK(8);
    sor.setStddevMulThresh(2.5);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(
      new pcl::PointCloud<pcl::PointXYZI>());
    sor.filter(*filtered);
    if (filtered->size() < 100) filtered = finite;

    pcl::PointXYZI minimum;
    pcl::PointXYZI maximum;
    pcl::getMinMax3D(*finite, minimum, maximum);
    const Eigen::Vector4f plane = estimateGroundPlane(filtered, options);
    const double ground_z_at_origin = -plane.w() / plane.z();

    const double origin_x = std::floor(
      (static_cast<double>(minimum.x) - options.padding) / options.resolution) * options.resolution;
    const double origin_y = std::floor(
      (static_cast<double>(minimum.y) - options.padding) / options.resolution) * options.resolution;
    const double max_x = static_cast<double>(maximum.x) + options.padding;
    const double max_y = static_cast<double>(maximum.y) + options.padding;
    const std::size_t width = static_cast<std::size_t>(std::ceil((max_x - origin_x) / options.resolution));
    const std::size_t height = static_cast<std::size_t>(std::ceil((max_y - origin_y) / options.resolution));
    if (width == 0 || height == 0 || width * height > 100000000ULL)
      throw std::runtime_error("Projected map dimensions are invalid or too large");

    octomap::OcTree octree(options.resolution);
    octree.setProbHit(0.70);
    octree.setProbMiss(0.40);
    std::size_t obstacle_points = 0;
    std::size_t ground_points = 0;
    for (const auto &point : filtered->points)
    {
      const double relative_height = heightAbovePlane(point, plane);
      if (std::abs(relative_height) <= options.ground_band)
      {
        octree.updateNode(octomap::point3d(point.x, point.y, point.z), false, true);
        ++ground_points;
      }
      else if (relative_height >= options.obstacle_min_height &&
               relative_height <= options.obstacle_max_height)
      {
        octree.updateNode(octomap::point3d(point.x, point.y, point.z), true, true);
        ++obstacle_points;
      }
    }
    octree.updateInnerOccupancy();

    std::vector<std::uint8_t> free_mask(width * height, 0);
    std::vector<std::uint8_t> occupied_mask(width * height, 0);
    for (auto iterator = octree.begin_leafs(); iterator != octree.end_leafs(); ++iterator)
    {
      const int x = static_cast<int>(std::floor((iterator.getX() - origin_x) / options.resolution));
      const int y = static_cast<int>(std::floor((iterator.getY() - origin_y) / options.resolution));
      if (x < 0 || y < 0 || x >= static_cast<int>(width) || y >= static_cast<int>(height))
        continue;
      const std::size_t index = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * width;
      if (octree.isNodeOccupied(*iterator)) occupied_mask[index] = 1;
      else free_mask[index] = 1;
    }

    free_mask = dilate(
      free_mask, width, height,
      static_cast<int>(std::ceil(options.free_radius / options.resolution)));
    occupied_mask = dilate(
      occupied_mask, width, height,
      static_cast<int>(std::ceil(options.obstacle_radius / options.resolution)));
    std::vector<std::int8_t> grid(width * height, -1);
    std::size_t free_cells = 0;
    std::size_t occupied_cells = 0;
    for (std::size_t index = 0; index < grid.size(); ++index)
    {
      if (free_mask[index])
      {
        grid[index] = 0;
        ++free_cells;
      }
      if (occupied_mask[index])
      {
        if (grid[index] == 0) --free_cells;
        grid[index] = 100;
        ++occupied_cells;
      }
    }

    fs::create_directories(options.output_dir);
    const fs::path pgm_path = options.output_dir / "map.pgm";
    const fs::path yaml_path = options.output_dir / "map.yaml";
    const fs::path octomap_path = options.output_dir / "map.bt";
    const fs::path metadata_path = options.output_dir / "metadata.json";
    writePgm(pgm_path, grid, width, height);
    octree.writeBinary(octomap_path.string());

    {
      std::ofstream yaml(yaml_path);
      yaml << std::setprecision(12)
           << "image: map.pgm\n"
           << "mode: trinary\n"
           << "resolution: " << options.resolution << "\n"
           << "origin: [" << origin_x << ", " << origin_y << ", 0.0]\n"
           // 205-gray pixels encode unknown. Their occupancy probability is
           // 50/255 = 0.196078, so the standard 0.196 threshold preserves
           // them as unknown rather than incorrectly making them free.
           << "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n";
    }
    {
      std::ofstream json(metadata_path);
      json << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"source_pcd\": \"" << options.input.string() << "\",\n"
           << "  \"points\": " << raw->size() << ",\n"
           << "  \"filtered_points\": " << filtered->size() << ",\n"
           << "  \"ground_points\": " << ground_points << ",\n"
           << "  \"obstacle_points\": " << obstacle_points << ",\n"
           << "  \"xyz_min\": [" << minimum.x << ", " << minimum.y << ", " << minimum.z << "],\n"
           << "  \"xyz_max\": [" << maximum.x << ", " << maximum.y << ", " << maximum.z << "],\n"
           << "  \"ground_plane\": [" << plane.x() << ", " << plane.y() << ", "
           << plane.z() << ", " << plane.w() << "],\n"
           << "  \"ground_z_at_origin\": " << ground_z_at_origin << ",\n"
           << "  \"obstacle_height\": [" << options.obstacle_min_height << ", "
           << options.obstacle_max_height << "],\n"
           << "  \"resolution\": " << options.resolution << ",\n"
           << "  \"origin\": [" << origin_x << ", " << origin_y << "],\n"
           << "  \"width\": " << width << ",\n"
           << "  \"height\": " << height << ",\n"
           << "  \"free_cells\": " << free_cells << ",\n"
           << "  \"occupied_cells\": " << occupied_cells << "\n"
           << "}\n";
    }

    std::cout << std::fixed << std::setprecision(6)
              << "{\"ok\":true,\"yaml\":\"" << yaml_path.string()
              << "\",\"pgm\":\"" << pgm_path.string()
              << "\",\"metadata\":\"" << metadata_path.string()
              << "\",\"ground_z\":" << ground_z_at_origin
              << ",\"width\":" << width << ",\"height\":" << height << "}" << std::endl;
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "pcd_to_nav_map: " << error.what() << std::endl;
    return 1;
  }
}
