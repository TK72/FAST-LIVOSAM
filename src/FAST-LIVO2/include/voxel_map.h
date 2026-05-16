/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include <list>
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

static int voxel_plane_id = 0;

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
  bool Degeneracy_mitigation;
  double Degeneracy_Trans_thres;
  double Degeneracy_Rot_thres;
  double CAPACITY;
  double CAPACITY_FEATS_MAP;
} VoxelMapConfig;

typedef struct PointToPlane
{
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_;
  double d_;
  double eigen_value_;
  bool is_valid_;
  float dis_to_plane_;
} PointToPlane;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;
  VoxelPlane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar &pv);

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);
  VoxelOctoTree *Insert(const pointWithVar &pv);
};

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  int current_frame_id_ = 0;
  ros::Publisher voxel_map_pub_;
//  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;
    struct VoxelLocHash {
        size_t operator()(const VOXEL_LOCATION& v) const noexcept {
            size_t h1 = std::hash<int64_t>{}(v.x);
            size_t h2 = std::hash<int64_t>{}(v.y);
            size_t h3 = std::hash<int64_t>{}(v.z);
            // 简单合并：已经够用了（也可换成更稳健的混合）
            return h1 ^ (h2<<1) ^ (h3<<2);
        }
    };
    struct VoxelLocEq {
        bool operator()(const VOXEL_LOCATION& a, const VOXEL_LOCATION& b) const noexcept {
            return a.x==b.x && a.y==b.y && a.z==b.z;
        }
    };
// 仍然对外暴露“坐标 -> 指针”（兼容旧代码）
    std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*> voxel_map_;

// LRU 链表：头=最近使用，尾=最久未用
    using CacheList = std::list<std::pair<VOXEL_LOCATION, VoxelOctoTree*>>;
    using CacheIter = CacheList::iterator;
    CacheList voxel_map_cache_;

// LRU 索引：坐标 -> 链表迭代器
    std::unordered_map<VOXEL_LOCATION, CacheIter> lru_index_;

    std::atomic<uint64_t> stat_voxel_created_{0};  // 新建的体素总数
    std::atomic<uint64_t> stat_voxel_evicted_{0};  // 被淘汰（delete）的体素总数
    uint64_t last_evicted_ = 0;                    // 最近一次 UpdateVoxelMap 删除的数量

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_;
  V3D extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  StatesGroup state_;
  V3D position_last_;

  V3D last_slide_position = {0,0,0};

  geometry_msgs::Quaternion geoQuat_;

  int feats_down_size_;
  int effct_feat_num_;
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

    struct DegeneracyResult
    {
        bool any = false;

        // 是否退化（按轴对齐后的 roll pitch yaw, x y z）
        std::array<bool,3> rot_degen{false,false,false};    // [roll pitch yaw]
        std::array<bool,3> trans_degen{false,false,false};  // [x y z]

        // 对齐到轴后的特征值（你已经写到了文件里，这里存下来供后续用）
        Eigen::Vector3d lam_trans_axis = Eigen::Vector3d::Zero(); // [λx λy λz]
        Eigen::Vector3d lam_rot_axis   = Eigen::Vector3d::Zero(); // [λroll λpitch λyaw]

        // 最大特征值（按块）
        double lamT_max = 0.0;
        double lamR_max = 0.0;

        // “退化程度”：ratio = λmax / max(λaxis, eps)
        Eigen::Vector3d ratio_trans = Eigen::Vector3d::Ones();
        Eigen::Vector3d ratio_rot   = Eigen::Vector3d::Ones();

        // “协方差膨胀因子”：inflate = ratio^2（你图 (32) 直接用这个）
        Eigen::Vector3d inflate_trans = Eigen::Vector3d::Ones();
        Eigen::Vector3d inflate_rot   = Eigen::Vector3d::Ones();

        // 你原来的指标也保留
        double kappa_trans = 1.0;
        double kappa_rot   = 1.0;
    };

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat,
                         double last_lio_update_time,
                         DegeneracyResult& degen);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void Degeneracy_detection(const MD(6,6)& H_T_H,double t,
    double kappa_thr_trans,   // 你在 MATLAB 得到的“平移分界阈值”
    double kappa_thr_rot,     // 你在 MATLAB 得到的“旋转分界阈值”
    DegeneracyResult& out);

  void suppressByEigMask(const Eigen::Matrix<double,6,6>& HTHdeal,
            const DegeneracyResult& degen,
            Eigen::Ref<Eigen::Matrix<double,6,1>> dx,
            double t);

  struct AxisEig {
      Eigen::Vector3d lam_axis;   // [lam_x, lam_y, lam_z]
      Eigen::Matrix3d U_axis;     // 列为对应轴的本征向量 [u_x | u_y | u_z]
  };

  AxisEig align_eigs_to_axes(const Eigen::Vector3d& lam,
                             const Eigen::Matrix3d& U,
                             const Eigen::Vector3d& ex,
                             const Eigen::Vector3d& ey,
                             const Eigen::Vector3d& ez);

//  struct VoxelLocHash {
//        size_t operator()(const VOXEL_LOCATION& v) const noexcept {
//            size_t h1 = std::hash<int64_t>{}(v.x);
//            size_t h2 = std::hash<int64_t>{}(v.y);
//            size_t h3 = std::hash<int64_t>{}(v.z);
//            return h1 ^ (h2<<1) ^ (h3<<2);
//        }
//  };
//  struct VoxelLocEq {
//        bool operator()(const VOXEL_LOCATION& a, const VOXEL_LOCATION& b) const noexcept {
//            return a.x==b.x && a.y==b.y && a.z==b.z;
//        }
//  };
//
//  using CacheList = std::list<std::pair<VOXEL_LOCATION, VoxelOctoTree*>>;
//  using CacheIter = CacheList::iterator;
//
//  CacheList voxel_map_cache_;  // 头=最近使用，尾=最久未用
//  std::unordered_map<VOXEL_LOCATION, CacheIter> lru_index_;

  void BuildVoxelMap();
  V3F RGBFromVoxel(const V3D &input_point);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);

  void pubVoxelMap();

  void mapSliding();
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

  void clearAll();
private:
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_