// export_map.cpp — dump a GLIM offline map to PLY for the cross-lidar align stage.
//
// Loads a GLIM `dump_path` (as written by `glim_rosbag ... -p dump_path:=DUMP`), captures
// each submap via the on_insert_submap callback fired during load, and writes into OUT:
//   OUT/map.ply                merged WORLD-frame cloud   (tier-1 registration input)
//   OUT/submaps/<id:06>.ply    each submap's points in its LOCAL frame (tier-2)
//   OUT/submaps/poses.txt      one row per submap: `id stamp x y z qx qy qz qw` (optimized)
//
// API pinned to glim @ 1007fec / gtsam_points @ 9d32e7d:
//   - GlobalMappingCallbacks::on_insert_submap is a CallbackSlot<void(const SubMap::ConstPtr&)>;
//     load() fires it once per submap, which is how we iterate the (private) submap list.
//   - SubMap::T_world_origin (Eigen::Isometry3d) is the optimized world pose; SubMap::frame
//     (gtsam_points::PointCloud::Ptr) holds LOCAL-frame points (world = T_world_origin * p);
//     origin_frame()->stamp is a representative timestamp.
// No PLY writer exists upstream, so we emit ASCII PLY by hand. Keep this file in sync with
// docker/slam-glim/map_export/src/export_map.cpp (separate Docker build contexts).

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <glim/mapping/callbacks.hpp>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/sub_map.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/util/callback_slot.hpp>
#include <gtsam_points/types/point_cloud.hpp>

namespace {

void write_ply(const std::string& path, const std::vector<Eigen::Vector3d>& pts) {
  std::ofstream ofs(path, std::ios::binary);
  ofs << "ply\nformat ascii 1.0\n"
      << "element vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "end_header\n";
  ofs << std::setprecision(6) << std::fixed;
  for (const auto& p : pts) {
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: export_map <glim_dump_dir> <out_dir>\n");
    return 64;
  }
  const std::string dump = argv[1];
  const std::string out = argv[2];

  std::vector<glim::SubMap::ConstPtr> submaps;
  glim::GlobalMappingCallbacks::on_insert_submap.add(
      [&](const glim::SubMap::ConstPtr& s) { submaps.push_back(s); });

  glim::GlobalMapping gm;  // default params suffice to load + read poses/points (CPU build)
  if (!gm.load(dump)) {
    std::fprintf(stderr, "export_map: failed to load GLIM dump: %s\n", dump.c_str());
    return 66;
  }
  if (submaps.empty()) {
    std::fprintf(stderr, "export_map: no submaps in dump: %s\n", dump.c_str());
    return 65;
  }

  const std::string sub_dir = out + "/submaps";
  std::filesystem::create_directories(sub_dir);

  std::ofstream poses(sub_dir + "/poses.txt");
  poses << "# id stamp x y z qx qy qz qw\n";
  poses << std::setprecision(9) << std::fixed;

  std::vector<Eigen::Vector3d> world;  // the merged map
  for (const auto& sm : submaps) {
    const Eigen::Isometry3d T = sm->T_world_origin;
    const auto& f = sm->frame;

    std::vector<Eigen::Vector3d> local;
    if (f) {
      local.reserve(f->size());
      for (std::size_t i = 0; i < f->size(); ++i) {
        const Eigen::Vector4d& p = f->points[i];
        local.emplace_back(p.x(), p.y(), p.z());
        const Eigen::Vector4d pw = T * p;
        world.emplace_back(pw.x(), pw.y(), pw.z());
      }
    }
    char name[64];
    std::snprintf(name, sizeof(name), "/%06d.ply", sm->id);
    write_ply(sub_dir + name, local);

    const Eigen::Vector3d t = T.translation();
    const Eigen::Quaterniond q(T.rotation());
    const double stamp = sm->frames.empty() ? 0.0 : sm->origin_frame()->stamp;
    poses << sm->id << ' ' << stamp << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
          << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
  }
  poses.close();

  write_ply(out + "/map.ply", world);
  std::printf("export_map: %zu submaps -> %s/map.ply (+ submaps/)\n", submaps.size(),
              out.c_str());
  return 0;
}
