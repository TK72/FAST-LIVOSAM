/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>
#include <list>
#include <unordered_set>
#include <cmath>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  // NEW: 每个点的权重（深度等形成的 per-point weight）
  std::vector<float> weights;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
    weights.reserve(SIZE_LARGE); // NEW
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
    weights.clear(); // NEW
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;

  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;

    // ------- LRU for visual feat_map -------
    using FeatCacheList = std::list<std::pair<VOXEL_LOCATION, VOXEL_POINTS*>>;
    FeatCacheList feat_cache_; // [head: 最近使用 / tail: 最久未用]
    std::unordered_map<VOXEL_LOCATION, FeatCacheList::iterator> feat_index_; // pos -> list it


// LRU 参数（已按你给定：voxel=1m, cap=2000；低水位 = 1800）
    size_t FEAT_CAP_HIGH_; // 体素条目上限
    size_t FEAT_CAP_LOW_; // 低水位，回落到这个数就停
    double FEAT_VOXEL_SIZE_;
    double FEAT_KEEP_RADIUS_M_ = 8.0; // 周围立方体半径（米）
    double FEAT_KEEP_AHEAD_M_ = 10.0; // 前向加深（米）
    double FEAT_KEEP_BACK_M_ = 4.0; // 后向保留（米）
    int FEAT_KEEP_Z_ = 1; // 垂直方向保留层数（±FEAT_KEEP_Z_）
    size_t FEAT_CAPACITY    = 2000;     // 你的上限

// 统计
    size_t last_evict_voxels_ = 0;
    size_t last_evict_points_ = 0;

// LRU 内部工具（在 vio.cpp 中实现）
    void touchFeatLRU_(const VOXEL_LOCATION& pos, VOXEL_POINTS* ptr);
    void evictFeatLRU_(const Eigen::Vector3d& cam_pos_W,
                       const Eigen::Vector3d& forward_W);
    void evictFeatLRU_Oldest_(size_t high_cap, size_t low_cap);

    using FeatLRUList = std::list<std::pair<VOXEL_LOCATION, VOXEL_POINTS*>>;
    FeatLRUList feat_lru_;
    std::unordered_map<VOXEL_LOCATION, FeatLRUList::iterator, VoxelMapManager::VoxelLocHash, VoxelMapManager::VoxelLocEq> feat_lru_index_;

    std::vector<VOXEL_POINTS*> feat_garbage_;
    std::unordered_set<VOXEL_POINTS*> feat_garbage_set_;
// 统一释放（只在“安全点”调用）
    void drainFeatGarbage();

    //半稠密-稀疏视觉地图
    std::vector<std::unique_ptr<VisualPoint>> semidense_pool_;   // 本帧新建的半稠密点（临时所有权）
    std::vector<VisualPoint*>                 semidense_pending_; // 待入图（下一步转给 feat_map）
    std::vector<cv::Point> semidense_candidates_;
    // 半稠密开关
    bool semidense_en;
    // --- 视觉鲁棒项与选择策略参数（默认值够用，可按需要重载） ---
    double huber_delta_; // Huber δ（像素灰度单位）
    double depth_max_m_; // 超过这深度的点直接丢弃（限深）
    double depth_ref_m_;  // 计算深度权重的参考深度（越近权重越大）
    int max_points_cap_; // 本帧用于优化的点上限
    int topk_total_;
    int topk_per_tile_;
    double min_grad_;
    double min_depth_m_;
    void pruneVisualSubmap_(const cv::Mat& img);
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_
