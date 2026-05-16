/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"

int  updateKdtreeCount = 0;   // 定义+初始化
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

double last_timestamp_gnss = -1.0 ;
geometry_msgs::PoseStamped msg_gnss_pose;
bool gnss_inited = false ;                        //  是否完成gnss初始化
shared_ptr<GnssProcess> p_gnss{new GnssProcess()};
GnssProcess gnss_data;
ros::Publisher pubGnssPath;
ros::Publisher pubPathUpdate;
nav_msgs::Path gps_path;
nav_msgs::Path globalPath;

vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 历史所有关键帧的角点集合（降采样）
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）

pcl::PointCloud<PointTypePose>::Ptr gnss_cloudKeyPoses6D{new pcl::PointCloud<PointTypePose>()};
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D{new pcl::PointCloud<PointType>()};         // 历史关键帧位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D{new pcl::PointCloud<PointTypePose>()}; // 历史关键帧位姿 (位置、姿态、时间戳）
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
float transformTobeMapped[6];
double last_lio_update_time = -1.0;   // 未就绪
/*loop closure*/
bool loopClosureEnableFlag;
float loopClosureFrequency; //   回环检测频率
int surroundingKeyframeSize;
float historyKeyframeSearchRadius;   // 回环检测 radius kdtree搜索半径
float historyKeyframeSearchTimeDiff; //  帧间时间阈值
int historyKeyframeSearchNum;        //   回环时多少个keyframe拼成submap
float historyKeyframeFitnessScore;   // icp 匹配阈值
bool potentialLoopFlag = false;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer;
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;
std::mutex mtx;
std::mutex mtxLoopInfo;
ros::Publisher pubHistoryKeyFrames;
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubCloudRegisteredRaw;
ros::Publisher pubLoopConstraintEdge;
ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOptimizedGlobalMap;

pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());
const string RESULT_PATH_TRAJ = std::string(ROOT_DIR) + "Log/result/integrated_to_init.txt";

static inline void AccumulateDegeneracyWorst(
        const VoxelMapManager::DegeneracyResult& cur,
        VoxelMapManager::DegeneracyResult& acc)
{
    acc.any = acc.any || cur.any;

    // 逐轴取“更坏”的膨胀因子（max）
    acc.inflate_rot   = acc.inflate_rot.cwiseMax(cur.inflate_rot);
    acc.inflate_trans = acc.inflate_trans.cwiseMax(cur.inflate_trans);

    // 退化标志取 OR
    for(int i=0;i<3;++i){
        acc.rot_degen[i]   = acc.rot_degen[i]   || cur.rot_degen[i];
        acc.trans_degen[i] = acc.trans_degen[i] || cur.trans_degen[i];
    }
}

gtsam::noiseModel::Diagonal::shared_ptr LIVMapper::MakeOdomNoiseFromDegeneracy(const VoxelMapManager::DegeneracyResult& d)  const
{
    // 1) 你的“基础里程计方差”（没退化时）
    //    你可以用常数；或者用前端传播出来的 outsigma（对角）作为 base。
    //    这里先给常数示例：旋转 1e-3 rad, 平移 5cm
    const double sig_rot   = 1e-3;  // sqrt(1e-6)  ≈ 0.057°
    const double sig_pos = 1e-2;  // sqrt(1e-4)  = 1 cm

    Eigen::Matrix<double,6,1> var;
    var << sig_rot*sig_rot, sig_rot*sig_rot, sig_rot*sig_rot,
            sig_pos*sig_pos, sig_pos*sig_pos, sig_pos*sig_pos;

    if (d.any) {
        for(int i=0;i<3;++i){
            if (d.rot_degen[i])   var(i)   *= 1;
            if (d.trans_degen[i]) var(3+i) *= 1;
        }
    }

    // 3) 防爆：上下限（非常重要！）
    const double min_var_rot = 1e-12;   // rad^2
    const double min_var_pos = 1e-10;   // m^2
    const double max_var_rot = 1e2;     // rad^2
    const double max_var_pos = 1e6;     // m^2

    for(int i=0;i<3;++i){
        var(i)   = std::min(std::max(var(i),   min_var_rot), max_var_rot);
        var(3+i) = std::min(std::max(var(3+i), min_var_pos), max_var_pos);
    }

    gtsam::Vector6 v;
    v << var(0),var(1),var(2),var(3),var(4),var(5);
    return gtsam::noiseModel::Diagonal::Variances(v);
}

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_undistort_pre.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");
  nh.param<bool>("common/imuAngularIsDegree", imuAngularIsDegree, false);
  nh.param<bool>("common/isPengyudata", isPengyudata, false);

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  nh.param<string>("common/gnss_topic", gnss_topic,"/gps/fix");
  nh.param<vector<double>>("extrin_calib/extrinR_Gnss2Lidar", extrinR_Gnss2Lidar, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinT_Gnss2Lidar", extrinT_Gnss2Lidar, vector<double>());
  Gnss_T_wrt_Lidar<<VEC_FROM_ARRAY(extrinT_Gnss2Lidar);
  Gnss_R_wrt_Lidar<<MAT_FROM_ARRAY(extrinR_Gnss2Lidar);
  nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);
  nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
  nh.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
  nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);
  nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

  nh.param<float>("gpsCovThreshold", gpsCovThreshold, 2.0);
  nh.param<float>("poseCovThreshold", poseCovThreshold, 25.0);
  nh.param<bool>("useGpsElevation", useGpsElevation, false);
  nh.param<bool>("recontructKdTree", recontructKdTree, false);
  // Visualization
  nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
  nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
  nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

  // loop clousre
  nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
  nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);
  nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);
  nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
  nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
  nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
  nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

  //semi dense parameters
  nh.param<bool>("vio/semidense_en", semidense_en, false);
  nh.param<double>("vio/huber_delta", huber_delta_, 10.0);
  nh.param<double>("vio/depth_max_m", depth_max_m_, 25.0);
  nh.param<double>("vio/depth_ref_m", depth_ref_m_, 8.0);
  nh.param<int>("vio/max_points_cap", max_points_cap_, 1000);
  nh.param<int>("vio/topk_total", topk_total_, 300);
  nh.param<int>("vio/topk_per_tile", topk_per_tile_, 2);
  nh.param<double>("vio/min_grad", min_grad_, 20.0);
  nh.param<double>("vio/min_depth_m", min_depth_m_, 0.3);

  //publish
  nh.param<bool>("publish/path_en", path_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam))
      throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->FEAT_VOXEL_SIZE_= voxelmap_manager->config_setting_.max_voxel_size_;
  vio_manager->FEAT_CAP_HIGH_= voxelmap_manager->config_setting_.CAPACITY_FEATS_MAP;
  vio_manager->FEAT_CAP_LOW_= vio_manager->FEAT_CAP_HIGH_ * 0.9;
  vio_manager->initializeVIO();

  //semi dense visual parameters
  vio_manager->semidense_en = semidense_en;
  vio_manager->huber_delta_= huber_delta_;
  vio_manager->depth_max_m_ = depth_max_m_;
  vio_manager->depth_ref_m_ = depth_ref_m_;
  vio_manager->max_points_cap_ = max_points_cap_;

  //high value points
  vio_manager->topk_total_= topk_total_;
  vio_manager->topk_per_tile_ = topk_per_tile_;
  vio_manager->min_grad_ = min_grad_;
  vio_manager->min_depth_m_ = min_depth_m_;

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) 
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);
  sub_gnss = nh.subscribe(gnss_topic, 200000, &LIVMapper::gnss_cbk, this);

  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 1);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 1);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 1);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage = it.advertise("/rgb_img", 1);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);

  pubPathUpdate = nh.advertise<nav_msgs::Path>("/path_update", 100000);                   //  isam更新后的path
  pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);
  pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/keyframe_submap", 1); // 发布局部关键帧map的特征点云
  pubOptimizedGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("/map_global_optimized", 1); // 发布局部关键帧map的特征点云

  // loop clousre
  pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/icp_loop_closure_history_cloud", 1);
  // loop point cloud
  pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/icp_loop_closure_corrected_cloud", 1);
  // loop edge
  pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);

}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() 
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  auto& last = LidarMeasures.measures.back();
  vio_manager->processFrame(last.img, _pv_list, voxelmap_manager->voxel_map_,
                            LidarMeasures.last_lio_update_time - _first_lidar_time);
  publish_img_rgb(pubImage, vio_manager->img_cp, ros::Time::now(), "camera_init");

  if (imu_prop_enable)
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO() 
{    
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;

  if (!feats_undistort || feats_undistort->empty())
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat, LidarMeasures.last_lio_update_time, curr_degen_);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
    double t5 = omp_get_wtime();

    if (!degen_between_kf_inited_) {
        degen_between_kf_ = curr_degen_;
        degen_between_kf_inited_ = true;
    } else {
        AccumulateDegeneracyWorst(curr_degen_, degen_between_kf_);
    }
  //从这里开始后端的因子图优化处理过程
  getCurPose();
  saveKeyFramesAndFactor();
  correctPoses();
  /******* Publish points *******/
  if (path_en){
      publish_path(pubPath);
      publish_gnss_path(pubGnssPath);                        //   发布gnss轨迹
      publish_path_update(pubPathUpdate);             //   发布经过isam2优化后的路径
  }
    double t6 = omp_get_wtime();

  //这里开始是原来fastlivo2原来发布里程计位姿的步骤
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  aver_time_fgo = aver_time_fgo * (frame_num - 1) / frame_num + (t6 - t5) / frame_num;

//   aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

    printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
           "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, fgo: %0.6f secs, total: %0.6f secs.\033[0m\n",
           t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_fgo, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::getCurPose()
{
    euler_cur = RotMtoEuler(_state.rot_end);
    transformTobeMapped[0] = euler_cur(0);                //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = euler_cur(1);                //  pitch
    transformTobeMapped[2] = euler_cur(2);                //  yaw
    transformTobeMapped[3] = _state.pos_end(0);          //  x
    transformTobeMapped[4] = _state.pos_end(1);          //   y
    transformTobeMapped[5] = _state.pos_end(2);          // z
}

void LIVMapper::saveKeyFramesAndFactor()
{
    //  计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (saveFrame() == false)
        return;

    std::cout << "[OdomNoise] inflR=" << degen_between_kf_.inflate_rot.transpose()
              << " inflT=" << degen_between_kf_.inflate_trans.transpose()
              << " any=" << degen_between_kf_.any << std::endl;

    // 激光里程计因子(from fast-livo),  输入的是frame_relative pose  帧间位姿(body 系下)
    addOdomFactor();
    //reset degeneracy count
    degen_between_kf_inited_ = false;
    // GPS因子 (UTM -> WGS84)
    addGPSFactor();
    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();

//    auto fmt = gtsam::DefaultKeyFormatter;
//    ROS_INFO_STREAM("graph.size=" << gtSAMgraph.size()
//                                  << "  init.size=" << initialEstimate.size());
//    gtSAMgraph.print("graph", fmt);
//    initialEstimate.print("init", fmt);
//
//    std::cout << "==== ISAM2 ====\n";
//    isam->print("isam");
//    isam->printStats();
//    isam->getFactorsUnsafe().print("isam factors", fmt);

    // 执行优化
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (aLoopIsClosed == true) // 有回环因子，多update几次
    {
        isam->update();
        isam->update();
        isam->update();
//        isam->update();
//        isam->update();
    }
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    // 优化结果
    isamCurrentEstimate = isam->calculateBestEstimate();
    // 当前帧位姿结果
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);

    // cloudKeyPoses3D加入当前帧位置
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    // 索引
    thisPose3D.intensity = cloudKeyPoses3D->size(); //  使用intensity作为该帧点云的index
    cloudKeyPoses3D->push_back(thisPose3D);

    // cloudKeyPoses6D加入当前帧位姿
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = LidarMeasures.last_lio_update_time;
    cloudKeyPoses6D->push_back(thisPose6D);

    // 位姿协方差
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // ESKF状态和方差  更新
    //state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    //Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());
    Eigen::Matrix3d Rot = EulerToRotM(Eigen::Vector3d(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw()));  // R = Rz(yaw)*Ry(pitch)*Rx(roll)
    //  更新状态量
    _state.pos_end(0) = pos(0);
    _state.pos_end(1) = pos(1);
    _state.pos_end(2) = pos(2);
    _state.rot_end = Rot;

    // 当前帧激光角点、平面点，降采样集合
    // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    // pcl::copyPointCloud(*feats_undistort,  *thisCornerKeyFrame);
    pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // 存储关键帧,没有降采样的点云

    // 保存特征点降采样集合
    // cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    updatePath(thisPose6D); //  可视化update后的path
}

bool LIVMapper::saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    // 前一帧位姿
    Eigen::Affine3f transStart = LIVMapper::pclPointToAffine3f(cloudKeyPoses6D->back());
    // 当前帧位姿
    Eigen::Affine3f transFinal = LIVMapper::trans2Affine3f(transformTobeMapped);

    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿

    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;
    return true;
}

void LIVMapper::addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // 第一帧先验
        auto priorNoise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(
                0, LIVMapper::trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, LIVMapper::trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        // —— 自适应噪声：根据 curr_degen_ 放大退化自由度的方差 ——
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
                MakeOdomNoiseFromDegeneracy(degen_between_kf_);

        gtsam::Pose3 poseFrom = LIVMapper::pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo   = LIVMapper::trans2gtsamPose(transformTobeMapped);

        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(),
                poseFrom.between(poseTo), odometryNoise));

        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

gtsam::Pose3 LIVMapper::pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

gtsam::Pose3 LIVMapper::trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

Eigen::Affine3f LIVMapper::pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

Eigen::Affine3f LIVMapper::trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}
void LIVMapper::correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // 清空里程计轨迹
        globalPath.poses.clear();
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D->points[i]);
        }
        ROS_INFO("ISMA2 Update");
        aLoopIsClosed = false;
    }
}

void LIVMapper::updatePath(const PointTypePose &pose_in)
{
    string odometryFrame = "camera_init";
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  std::remove(RESULT_PATH_TRAJ.c_str());
  ros::Rate rate(5000);

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new gtsam::ISAM2(parameters);

  //回环线程
  startLoopThread();
  while (ros::ok()) 
  {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }

  stopLoopThread();
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}


void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    if (!imu_en) return;
    if (last_timestamp_lidar < 0.0) return;

    // ===== [A] 原始时间戳 & offset 前后对比（放这里：复制 msg 后，减 offset 前后）=====
    const double t_raw = msg_in->header.stamp.toSec();

    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
    double timestamp = msg->header.stamp.toSec();
    const double dt_lidar_before_round = last_timestamp_lidar - timestamp;

    ROS_INFO_THROTTLE(1.0,
                      "[IMU] raw=%.6f offset=%.6f corrected=%.6f last_lidar=%.6f dt_lidar=%.6f ros_fix=%d",
                      t_raw, imu_time_offset, timestamp, last_timestamp_lidar, dt_lidar_before_round, (int)ros_driver_fix_en);

    if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
    {
        ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
    }

    // ===== [B] round 修正是否触发（放这里：round 前后对比）=====
    double add_round = 0.0;
    if (ros_driver_fix_en) {
        add_round = std::round(last_timestamp_lidar - timestamp);
        timestamp += add_round;
    }
    msg->header.stamp = ros::Time().fromSec(timestamp);

    ROS_INFO_THROTTLE(1.0,
                      "[IMU] after_round=%.6f add_round=%.3f dt_lidar_after=%.6f",
                      timestamp, add_round, last_timestamp_lidar - timestamp);

    // ===== [C] 回退检测：打印更多上下文（放这里：进入 lock 后、判断前）=====
    mtx_buffer.lock();

    if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
    {
        const double back = last_timestamp_imu - timestamp;
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        ROS_ERROR("[IMU] loop back! back=%.6f last_imu=%.6f cur=%.6f raw=%.6f offset=%.6f",
                  back, last_timestamp_imu, timestamp, t_raw, imu_time_offset);
        return;
    }

    // ===== [D] 检查 IMU 频率/间隔（放这里：更新 last_timestamp_imu 前后都行）=====
    if (last_timestamp_imu > 0.0) {
        const double dt_imu = timestamp - last_timestamp_imu;
        ROS_INFO_THROTTLE(1.0, "[IMU] dt_imu=%.6f (Hz≈%.1f) buf=%zu",
                          dt_imu, (dt_imu>1e-6? 1.0/dt_imu : 0.0), imu_buffer.size());
    }

    last_timestamp_imu = timestamp;

    // ===== [E] 角速度单位转换前后（放这里：读取 gyr 后、写回前后）=====
    Eigen::Vector3d gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

    if(!imuAngularIsDegree) {
        // rad/s
        msg->angular_velocity.x = gyr.x();
        msg->angular_velocity.y = gyr.y();
        msg->angular_velocity.z = gyr.z();
    } else {
        // deg/s -> rad/s
        msg->angular_velocity.x = gyr.x() * (M_PI / 180);
        msg->angular_velocity.y = gyr.y() * (M_PI / 180);
        msg->angular_velocity.z = gyr.z() * (M_PI / 180);
    }

    ROS_INFO_THROTTLE(1.0,
                      "[IMU] gyro_in=(%.5f %.5f %.5f) isDeg=%d gyro_used=(%.5f %.5f %.5f)",
                      gyr.x(), gyr.y(), gyr.z(), (int)imuAngularIsDegree,
                      msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();

    // 下面保持不变……
    if (imu_prop_enable)
    {
        mtx_buffer_imu_prop.lock();
        if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
        newest_imu = *msg;
        new_imu = true;
        mtx_buffer_imu_prop.unlock();
    }
    sig_buffer.notify_all();
}


//void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
//{
//  if (!imu_en) return;
//
//  if (last_timestamp_lidar < 0.0) return;
//  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
//  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
//  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
//  double timestamp = msg->header.stamp.toSec();
//
//  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
//  {
//    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
//  }
//
//  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
//  msg->header.stamp = ros::Time().fromSec(timestamp);
//
//  mtx_buffer.lock();
//
//  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
//  {
//    mtx_buffer.unlock();
//    sig_buffer.notify_all();
//    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
//    return;
//  }
//
//  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
//  // {
//
//  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
//  //   mtx_buffer.unlock();
//  //   sig_buffer.notify_all();
//  //   return;
//  // }
//
//  last_timestamp_imu = timestamp;
//
//  Eigen::Vector3d gyr(msg -> angular_velocity.x, msg -> angular_velocity.y, msg -> angular_velocity.z);
//  if(!imuAngularIsDegree) {
//      msg -> angular_velocity.x = gyr.x();
//      msg -> angular_velocity.y = gyr.y();
//      msg -> angular_velocity.z = gyr.z();
//  } else {
//      msg -> angular_velocity.x = gyr.x() * (M_PI / 180);
//      msg -> angular_velocity.y = gyr.y() * (M_PI / 180);
//      msg -> angular_velocity.z = gyr.z() * (M_PI / 180);
//  }
//
//  imu_buffer.push_back(msg);
//  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
//  mtx_buffer.unlock();
//  if (imu_prop_enable)
//  {
//    mtx_buffer_imu_prop.lock();
//    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
//    newest_imu = *msg;
//    new_imu = true;
//    mtx_buffer_imu_prop.unlock();
//  }
//  sig_buffer.notify_all();
//}

void LIVMapper::gnss_cbk(const nav_msgs::Odometry::ConstPtr& msg_in)
{
    //  ROS_INFO("GNSS DATA IN ");
    double timestamp = msg_in->header.stamp.toSec();

    mtx_buffer.lock();

    // 没有进行时间纠正
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec();
    gnss_data.status = 0;
    gnss_data.service = 1;
    gnss_data.pose_cov[0] = msg_in->pose.covariance[0];
    gnss_data.pose_cov[1] = msg_in->pose.covariance[7];
    gnss_data.pose_cov[2] = msg_in->pose.covariance[14];

    mtx_buffer.unlock();

    if(!gnss_inited){           //  初始化位置
        gnss_data.InitOriginPosition(msg_in->pose.pose.position.x, msg_in->pose.pose.position.y, msg_in->pose.pose.position.z) ;
        gnss_inited = true ;
    }else{
        if(isPengyudata) {
            gnss_data.UpdateXYZ_direct_give(msg_in->pose.pose.position.x, msg_in->pose.pose.position.y,
                                            msg_in->pose.pose.position.z);
        }
        else
            gnss_data.UpdateXYZ(msg_in->pose.pose.position.x, msg_in->pose.pose.position.y, msg_in->pose.pose.position.z) ;             //  WGS84 -> ENU  ???  调试结果好像是 NED 北东地

        Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();
        gnss_pose(0,3) = gnss_data.local_N ;                 //    北
        gnss_pose(1,3) = gnss_data.local_E ;                 //     东
        gnss_pose(2,3) = -gnss_data.local_U ;                 //    地

        Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
        gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
        gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下, （当前Gnss_T_wrt_Lidar，只是一个大致的初值）

        nav_msgs::Odometry gnss_data_enu ;
        // add new message to buffer:
        gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
        gnss_data_enu.pose.pose.position.x =  gnss_pose(0,3) ;  //gnss_data.local_E ;   北
        gnss_data_enu.pose.pose.position.y =  gnss_pose(1,3) ;  //gnss_data.local_N;    东
        gnss_data_enu.pose.pose.position.z =  gnss_pose(2,3) ;  //  地

        //在python脚本中已经将姿态转换为了LIDAR系到WORLD下(LIDAR在0时刻的坐标系）的旋转姿态
        auto q_original = msg_in->pose.pose.orientation;
        gnss_data_enu.pose.pose.orientation.x =  q_original.x;
        gnss_data_enu.pose.pose.orientation.y =  q_original.y;
        gnss_data_enu.pose.pose.orientation.z =  q_original.z;
        gnss_data_enu.pose.pose.orientation.w =  q_original.w;

        gnss_data_enu.pose.covariance[0] = msg_in->pose.covariance[0] ;
        gnss_data_enu.pose.covariance[7] = msg_in->pose.covariance[7];
        gnss_data_enu.pose.covariance[14] = msg_in->pose.covariance[14];
        gnss_data_enu.pose.covariance[21] = msg_in->pose.covariance[21] ;
        gnss_data_enu.pose.covariance[28] = msg_in->pose.covariance[28] ;
        gnss_data_enu.pose.covariance[35] = msg_in->pose.covariance[35] ;

        gnss_buffer.push_back(gnss_data_enu);

        // visial gnss path in rviz:
        msg_gnss_pose.header.frame_id = "camera_init";
        msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);

        msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;
        msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
        msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

        gps_path.poses.push_back(msg_gnss_pose);

        //  save_gnss path
        PointTypePose thisPose6D;
        thisPose6D.x = msg_gnss_pose.pose.position.x ;
        thisPose6D.y = msg_gnss_pose.pose.position.y ;
        thisPose6D.z = msg_gnss_pose.pose.position.z ;
        thisPose6D.intensity = 0;
        thisPose6D.roll =0;
        thisPose6D.pitch = 0;
        thisPose6D.yaw = 0;
        thisPose6D.time = LidarMeasures.last_lio_update_time;
        gnss_cloudKeyPoses6D->push_back(thisPose6D);
    }
}

void LIVMapper::addGPSFactor()
{
    if (gnss_buffer.empty())
        return;
    // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
    if (cloudKeyPoses3D->points.empty() || cloudKeyPoses3D->points.size() == 1)
        return;
    static PointType lastGPSPoint;      // 最新的gps数据
    while (!gnss_buffer.empty())
    {
        // 删除当前帧0.2s之前的里程计
        if (gnss_buffer.front().header.stamp.toSec() < LidarMeasures.last_lio_update_time - 0.05)
        {
            gnss_buffer.pop_front();
        }
            // 超过当前帧0.2s之后，退出
        else if (gnss_buffer.front().header.stamp.toSec() > LidarMeasures.last_lio_update_time + 0.05)
        {
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();
            // GPS噪声协方差太大，不能用
            float noise_x = thisGPS.pose.covariance[0];         //  x 方向的协方差
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];      //   z(高层)方向的协方差
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                continue;
            // GPS里程计位置
            float gps_x = thisGPS.pose.pose.position.x;
            float gps_y = thisGPS.pose.pose.position.y;
            float gps_z = thisGPS.pose.pose.position.z;
            if (!useGpsElevation)           //  是否使用gps的高度
            {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }

            // (0,0,0)无效数据
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                continue;
            // 每隔5m添加一个GPS里程计
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pcl::squaredEuclideanDistance(curGPSPoint, lastGPSPoint) < 25.0)
                continue;
            else
                lastGPSPoint = curGPSPoint;
            // 添加GPS因子
            gtsam::Vector Vector3(3);
            Vector3 << max(noise_x, 0.1f), max(noise_y, 0.1f), max(noise_z, 0.1f);
            //Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
            gtSAMgraph.add(gps_factor);
            aLoopIsClosed = true;
            ROS_INFO("GPS Factor Added");
            break;
        }
    }
}

void LIVMapper::addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first; //   cur
        int indexTo = loopIndexQueue[i].second;  //    pre
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = img_time_buffer.front() + exposure_time_init;

        if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();

        double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
        double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

        if (img_capture_time < meas.last_lio_update_time + 0.00001)
        {
          img_buffer.pop_front();
          img_time_buffer.pop_front();
          ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
          return false;
        }

        if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
        {
            // ROS_ERROR("lost first camera frame");
            // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
            // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
            return false;
        }
        // 完全复用读取LIDAR点云的代码！
        struct MeasureGroup m;
        m.imu.clear();
        m.lio_time = img_capture_time;
        mtx_buffer.lock();
        while (!imu_buffer.empty())
        {
            if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

            if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

            imu_buffer.pop_front();
            // printf("[ Data Cut ] imu time: %lf \n",
            // imu_buffer.front()->header.stamp.toSec());
        }
        mtx_buffer.unlock();
        sig_buffer.notify_all();

        *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
        PointCloudXYZI().swap(*meas.pcl_proc_next);

        int lid_frame_num = lid_raw_data_buffer.size();
        int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
        meas.pcl_proc_cur->reserve(max_size);
        meas.pcl_proc_next->reserve(max_size);
        // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

        while (!lid_raw_data_buffer.empty())
        {
            if (lid_header_time_buffer.front() > img_capture_time) break;
            auto pcl(lid_raw_data_buffer.front()->points);
            double frame_header_time(lid_header_time_buffer.front());
            float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

            for (int i = 0; i < pcl.size(); i++)
            {
                auto pt = pcl[i];
                if (pcl[i].curvature < max_offs_time_ms)
                {
                    pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
                    meas.pcl_proc_cur->points.push_back(pt);
                }
                else
                {
                    pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
                    meas.pcl_proc_next->points.push_back(pt);
                }
            }
            lid_raw_data_buffer.pop_front();
            lid_header_time_buffer.pop_front();
        }
        meas.measures.push_back(m);
        meas.lio_vio_flg = LIO;
        return true;
    }

    case LIO:
    {
      const double img_capture_time = img_time_buffer.front() + exposure_time_init;

      meas.lio_vio_flg = VIO;
      meas.measures.clear();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      img_buffer.pop_front();
      img_time_buffer.pop_front();

      mtx_buffer.unlock();
      sig_buffer.notify_all();

      meas.measures.push_back(m);
      lidar_pushed = false;
      return true;

    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

//void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
//{
//  cv::Mat img_rgb = vio_manager->img_cp;
//  cv_bridge::CvImage out_msg;
//  out_msg.header.stamp = ros::Time::now();
//  // out_msg.header.frame_id = "camera_init";
//  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
//  out_msg.image = img_rgb;
//  pubImage.publish(out_msg.toImageMsg());
//}

void LIVMapper::publish_img_rgb(const image_transport::Publisher& pub,
                     const cv::Mat& img_bgr,
                     const ros::Time& stamp,
                     const std::string& frame_id)
{
    if (img_bgr.empty() || pub.getNumSubscribers() == 0) return;
    cv_bridge::CvImage out;
    out.header.stamp = stamp;
    out.header.frame_id = frame_id;
    out.encoding = sensor_msgs::image_encodings::BGR8;
    out.image = img_bgr;
    pub.publish(out.toImageMsg());
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes,
                                    VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;

  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255;
          // else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255;
          // else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255;
          // else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en)
  {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else
  {
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";

    if (pubLaserCloudFullRes.getNumSubscribers() > 0) {   // 没有人订阅就别构造/发布巨型消息
        // buildBigCloudOnlyWhenNeeded();
        pubLaserCloudFullRes.publish(laserCloudmsg);
    }

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en)
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;

    if (img_en)
    {
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;

    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (img_en)
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub);
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

//void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
//{
//  if (pcl_w_wait_pub->empty()) return;
//  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
//  if (img_en)
//  {
//    static int pub_num = 1;
//    *pcl_wait_pub += *pcl_w_wait_pub;
//    if(pub_num == pub_scan_num)
//    {
//      pub_num = 1;
//      size_t size = pcl_wait_pub->points.size();
//      laserCloudWorldRGB->reserve(size);
//      // double inv_expo = _state.inv_expo_time;
//      cv::Mat img_rgb = vio_manager->img_rgb;
//      for (size_t i = 0; i < size; i++)
//      {
//        PointTypeRGB pointRGB;
//        pointRGB.x = pcl_wait_pub->points[i].x;
//        pointRGB.y = pcl_wait_pub->points[i].y;
//        pointRGB.z = pcl_wait_pub->points[i].z;
//
//        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
//        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
//        V2D pc(vio_manager->new_frame_->w2c(p_w));
//
//        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
//        {
//          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
//          pointRGB.r = pixel[2];
//          pointRGB.g = pixel[1];
//          pointRGB.b = pixel[0];
//          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
//          // if (pointRGB.r > 255) pointRGB.r = 255;
//          // else if (pointRGB.r < 0) pointRGB.r = 0;
//          // if (pointRGB.g > 255) pointRGB.g = 255;
//          // else if (pointRGB.g < 0) pointRGB.g = 0;
//          // if (pointRGB.b > 255) pointRGB.b = 255;
//          // else if (pointRGB.b < 0) pointRGB.b = 0;
//          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
//        }
//      }
//    }
//    else
//    {
//      pub_num++;
//    }
//  }
//
//  /*** Publish Frame ***/
//  sensor_msgs::PointCloud2 laserCloudmsg;
//  if (img_en)
//  {
//    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
//    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
//  }
//  else
//  {
//    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
//  }
//  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
//  laserCloudmsg.header.frame_id = "camera_init";
////  pubLaserCloudFullRes.publish(laserCloudmsg);
//    if (pubLaserCloudFullRes.getNumSubscribers() > 0) {   // 没有人订阅就别构造/发布巨型消息
//        // buildBigCloudOnlyWhenNeeded();
//        pubLaserCloudFullRes.publish(laserCloudmsg);
//    }
//
//  /**************** save map ****************/
//  /* 1. make sure you have enough memories
//  /* 2. noted that pcd save will influence the real-time performences **/
//  if (pcd_save_en)
//  {
//    int size = feats_undistort->points.size();
//    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
//    static int scan_wait_num = 0;
//
//    if (img_en)
//    {
//      *pcl_wait_save += *laserCloudWorldRGB;
//    }
//    else
//    {
//      *pcl_wait_save_intensity += *pcl_w_wait_pub;
//    }
//    scan_wait_num++;
//
//    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
//    {
//      pcd_index++;
//      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
//      pcl::PCDWriter pcd_writer;
//      if (pcd_save_en)
//      {
//        cout << "current scan saved to /PCD/" << all_points_dir << endl;
//        if (img_en)
//        {
//          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
//          PointCloudXYZRGB().swap(*pcl_wait_save);
//        }
//        else
//        {
//          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
//          PointCloudXYZI().swap(*pcl_wait_save_intensity);
//        }
//        Eigen::Quaterniond q(_state.rot_end);
//        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
//                     << " " << q.z() << " " << endl;
//        scan_wait_num = 0;
//      }
//    }
//  }
//  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub);
//  PointCloudXYZI().swap(*pcl_w_wait_pub);
//}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
//    pubSubVisualMap.publish(laserCloudmsg);
      if (pubSubVisualMap.getNumSubscribers() > 0) {   // 没有人订阅就别构造/发布巨型消息
          // buildBigCloudOnlyWhenNeeded();
          pubSubVisualMap.publish(laserCloudmsg);
      }
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
//  pubLaserCloudEffect.publish(laserCloudFullRes3);
    if (pubLaserCloudEffect.getNumSubscribers() > 0) {   // 没有人订阅就别构造/发布巨型消息
        // buildBigCloudOnlyWhenNeeded();
        pubLaserCloudEffect.publish(laserCloudFullRes3);
    }
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";

  //When performing dynamic objects removal,the time needs to be set here.
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);

  //When performing
//  ros::Time stamp;
//  stamp.fromSec(LidarMeasures.last_lio_update_time);   // double -> ros::Time
//  odomAftMapped.header.stamp = stamp;// ros::Time().fromSec(lidar_end_time);

  set_posestamp(odomAftMapped.pose.pose);
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);

  std::ofstream pose1(RESULT_PATH_TRAJ, std::ios::app);
  pose1.setf(std::ios::scientific, std::ios::floatfield);
  pose1.precision(12);
//            pose1<< Measures.lidar_beg_time - first_lidar_time<<
  pose1<< LidarMeasures.last_lio_update_time <<
         " "<< _state.pos_end(0)<<" "<<_state.pos_end(1)<<" "<<_state.pos_end(2)<<" "
         << geoQuat.x << " "
         << geoQuat.y << " "
         << geoQuat.z << " "
         << geoQuat.w <<std::endl;
  pose1.close();
}

void LIVMapper::startLoopThread() {
    if (loopthread_.joinable()) return;  // 已经在跑
    loop_stop_.store(false);
    loopthread_ = std::thread(&LIVMapper::loopClosureThread, this);
}

void LIVMapper::stopLoopThread() {
    loop_stop_.store(true);
    if (loopthread_.joinable()) loopthread_.join();
}

void LIVMapper::loopClosureThread()
{
    if (loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   回环频率
    while (ros::ok() && !loop_stop_.load())
    {
        performLoopClosure();   //  回环检测
        visualizeLoopClosure(); // rviz展示闭环边
        rate.sleep();
    }
}

void LIVMapper::performLoopClosure()
{
    double t = LidarMeasures.last_lio_update_time;
    if (!std::isfinite(t) || t < 0.0) {   // 不存在/未就绪/异常
        ROS_WARN_THROTTLE(5.0, "[loop] last_lio_update_time not set, fallback to 0");
        t = 0.0;
    }
    ros::Time timeLaserInfoStamp;
    timeLaserInfoStamp.fromSec(t);
    string odometryFrame = "camera_init";

    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
    {
        return;
    }

    // 提取
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //  cue keyframe
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //   history keyframe submap
    {
        // 提取当前关键帧特征点集合，降采样
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); //  将cur keyframe 转换到world系下
        // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); //  选取historyKeyframeSearchNum个keyframe拼成submap
        // 如果特征点较少，返回
        // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        //     return;
        // 发布闭环匹配关键帧局部map
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // scan-to-map，调用icp匹配
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // 未收敛，或者匹配不够好
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    std::cout << "icp success" << std::endl;

    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // 闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore() ; //  loop_clousre  noise from icp
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

    // 添加闭环因子需要的数据
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    loopIndexContainer[loopKeyCur] = loopKeyPre; //   使用hash map 存储回环对
}

void LIVMapper::visualizeLoopClosure()
{
    double t = LidarMeasures.last_lio_update_time;
    if (!std::isfinite(t) || t < 0.0) {   // 不存在/未就绪/异常
        ROS_WARN_THROTTLE(5.0, "[loop] last_lio_update_time not set, fallback to 0");
        t = 0.0;
    }
    ros::Time timeLaserInfoStamp;
    timeLaserInfoStamp.fromSec(t);
    string odometryFrame = "camera_init";

    if (loopIndexContainer.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}

void LIVMapper::publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(LidarMeasures.last_lio_update_time);
    gps_path.header.frame_id = "camera_init";

    static int gnss_path_count = 0;
    gnss_path_count++;
    if (gnss_path_count % 10 == 0)
    {
        pubPath.publish(gps_path);
    }
}

void LIVMapper::publish_path_update(const ros::Publisher pubPath)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(LidarMeasures.last_lio_update_time); //  时间戳
    string odometryFrame = "camera_init";
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int global_path_count = 0;
        global_path_count++;
        if (global_path_count % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}

//回环检测三大要素
// 1.设置最小时间差，太近没必要
// 2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
// 3.根据当前最小距离重新计算等待时间
bool LIVMapper::detectLoopClosureDistance(int *latestID, int *closestID)
{
    // 当前关键帧帧
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  当前关键帧索引
    int loopKeyPre = -1;

    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - LidarMeasures.last_lio_update_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    ROS_INFO("Find loop closure frame ");
    return true;
}

void LIVMapper::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;

        // *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); //  fast-lio 没有进行特征提取，默认点云就是surf
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
//    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
//    downSizeFilterICP.setInputCloud(nearKeyframes);
//    downSizeFilterICP.filter(*cloud_temp);
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloud_filtered2(new pcl::PCLPointCloud2());
    pcl::PointCloud<PointType>::Ptr  cloud_temp (new pcl::PointCloud<PointType>);
    pcl::toPCLPointCloud2(*nearKeyframes, *cloud2);
    //pcl::VoxelGrid<PointType> downSizeFilter2;
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor2;
    sor2.setInputCloud(cloud2);
    sor2.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    sor2.filter(*cloud_filtered2);
    pcl::fromPCLPointCloud2(*cloud_filtered2, *cloud_temp);
    //downSizeFilterICP.setInputCloud(cloud_temp);

    *nearKeyframes = *cloud_temp;
}

/**
 * 发布thisCloud，返回thisCloud对应msg格式
 */
sensor_msgs::PointCloud2 LIVMapper::publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

/**
 * 对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
 */
pcl::PointCloud<PointType>::Ptr LIVMapper::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // 注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释
    // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_b_lidar(extR);       //  获取  body2lidar  外参
    T_b_lidar.pretranslate(extT);

    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();

    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  转换矩阵

    Eigen::Isometry3d transCur = T_w_lidar;

#pragma omp parallel for num_threads(4)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}
