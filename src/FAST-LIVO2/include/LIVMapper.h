/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "vio.h"
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <vikit/camera_loader.h>
#include <GNSS_Processing.hpp>
#include <pcl/common/distances.h>
#include <pcl/registration/icp.h>
//gtsam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
//kdtree
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents();
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  void getCurPose();
//  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_img_rgb(const image_transport::Publisher& pub,
                       const cv::Mat& img_bgr,
                       const ros::Time& stamp,
                       const std::string& frame_id);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);
  void publish_gnss_path(const ros::Publisher pubPath);
  void publish_path_update(const ros::Publisher pubPath);
  void readParameters(ros::NodeHandle &nh);
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic, gnss_topic;
  bool imuAngularIsDegree;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<nav_msgs::Odometry> gnss_buffer;
  deque<double> img_time_buffer;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  vector<double> extrinT_Gnss2Lidar;
  vector<double> extrinR_Gnss2Lidar;
  M3D Gnss_R_wrt_Lidar;         // gnss  与 imu 的外参
  V3D Gnss_T_wrt_Lidar;
  bool isPengyudata;

  //gnss
  bool useImuHeadingInitialization;
  bool useGpsElevation;             //  是否使用gps高层优化
  float gpsCovThreshold;          //   gps方向角和高度差的协方差阈值
  float poseCovThreshold;       //  位姿协方差阈值  from isam2

  //gtsam(back end)
  void saveKeyFramesAndFactor();
  bool saveFrame();
  void addOdomFactor();
  void addGPSFactor();
  void addLoopFactor();
  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);
  gtsam::Pose3 trans2gtsamPose(float transformIn[]);
  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);
  Eigen::Affine3f trans2Affine3f(float transformIn[]);
  void gnss_cbk(const nav_msgs::Odometry::ConstPtr& msg_in);
  void correctPoses();
  void updatePath(const PointTypePose &pose_in);
  void recontructIKdTree();
  float surroundingkeyframeAddingDistThreshold;  //  判断是否为关键帧的距离阈值
  float surroundingkeyframeAddingAngleThreshold; //  判断是否为关键帧的角度阈值
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;
  bool recontructKdTree;
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;
  float mappingSurfLeafSize;
  //loopclosure
  void startLoopThread();
  void stopLoopThread();
  void performLoopClosure();
  void loopClosureThread();
  void visualizeLoopClosure();
  std::atomic<bool> loop_stop_{true};
  std::thread loopthread_;
  bool detectLoopClosureDistance(int *latestID, int *closestID);
  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum);
  sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame);
  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn);

    VoxelMapManager::DegeneracyResult curr_degen_;
    VoxelMapManager::DegeneracyResult degen_between_kf_;
    bool degen_between_kf_inited_ = false;

    gtsam::noiseModel::Diagonal::shared_ptr MakeOdomNoiseFromDegeneracy(const VoxelMapManager::DegeneracyResult& d) const;

  //publish
  bool path_en;

  double IMG_POINT_COV;
  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_undistort_pre;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;
  ofstream fout_pre, fout_out, fout_pcd_pos;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Subscriber sub_gnss;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  image_transport::Publisher pubImage;
  ros::Publisher mavros_pose_publisher;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  double aver_time_fgo = 0;

  // 半稠密开关
  bool semidense_en;
  double huber_delta_; // Huber δ（像素灰度单位）
  double depth_max_m_; // 超过这深度的点直接丢弃（限深）
  double depth_ref_m_;  // 计算深度权重的参考深度（越近权重越大）
  int max_points_cap_; // 本帧用于优化的点上限
  int topk_total_;
  int topk_per_tile_;
  double min_grad_;
  double min_depth_m_;

};
#endif
