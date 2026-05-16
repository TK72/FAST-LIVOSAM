/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/

// PCL specific includes
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include "pcl_ros/impl/transforms.hpp"

#include <ros/ros.h>
#include "center_pointpillars/centerpoint.h"

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <std_msgs/Header.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>

#include <algorithm>
#include <ros/package.h>

#include <deque>
#include <numeric>  // std::accumulate
#include <cmath>    // std::cos, std::sin, std::atan2, std::fabs

#include <mutex>
#include <thread>
#include <condition_variable>

#include <unordered_map>
#include <pcl/filters/filter.h>   // removeNaNFromPointCloud
#include "preprocess.h"           // velodyne_ros::Point + REGISTER
using PointT   = velodyne_ros::Point;
using CloudT   = pcl::PointCloud<PointT>;
using CloudTPtr = CloudT::Ptr;


#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

std::vector<unsigned char> color;

std::vector<double> avg_centerpoint_time;

static inline double wall_now_ms()
{
    // 用 WallTime：不受 /use_sim_time 影响，适合做性能统计
    return ros::WallTime::now().toSec() * 1000.0; // ms
}

static inline size_t packCloudToPillarsInput(
        const CloudT& cloud,
        float* h_points, size_t max_points,
        float min_range, float max_range)
{
    const float min2 = min_range * min_range;
    const float max2 = max_range * max_range;

    size_t n = 0;
    for (const auto& p : cloud.points) {
        const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
        if (r2 < min2 || r2 > max2) continue;
        if (n >= max_points) break;

        float* dst = h_points + n * 5;
        dst[0] = p.x;
        dst[1] = p.y;
        dst[2] = p.z;
        dst[3] = 0.0f;
        dst[4] = 0.0f;
        ++n;
    }
    return n;
}


static inline pcl::PointCloud<pcl::PointXYZI> toXYZI(const CloudT& in)
{
    pcl::PointCloud<pcl::PointXYZI> out;
    out.points.reserve(in.points.size());
    for (const auto& p : in.points) {
        pcl::PointXYZI q;
        q.x = p.x; q.y = p.y; q.z = p.z;
        q.intensity = p.intensity;
        out.points.push_back(q);
    }
    out.width = (uint32_t)out.points.size();
    out.height = 1;
    out.is_dense = in.is_dense;
    return out;
}

void GetDeviceInfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

void initDevice(int devNum) {
    int dev = devNum;
    cudaDeviceProp deviceProp;

    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    GPU_CHECK(cudaSetDevice(dev));
}

namespace cpp {

class Center_PointPillars_ROS {
  public:
    Center_PointPillars_ROS(ros::NodeHandle nh);
    ~Center_PointPillars_ROS();

    void Process();
    void extractBBoxPointcloud(std::vector<Bndbox> filter_BBox,
                               const CloudTPtr& cloud_in,
                               CloudTPtr& cloud_out,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster);

    struct FrameBoxes {
        ros::Time stamp;
        std::vector<Bndbox> boxes;
    };
    std::deque<FrameBoxes> box_hist_;
    struct Pose {
        ros::Time stamp;
        Eigen::Matrix4f T; // odom系位姿
    };
    std::deque<Pose> odom_hist_;

    double coast_time_sec_ = 0.8;     // 允许回放历史框的时间窗口
    int    coast_max_frames_ = 8;     // 最多回放的历史帧数
    double inflate_ratio_ = 0.12;     // 尺寸按比例膨胀
    double inflate_m_ = 0.25;         // 尺寸再加固定冗余（米）
    double merge_center_thresh_ = 1.0; // 历史框与当前框中心小于该距离则认为重复
    bool normalize_world_cloud_to_local_ = true;
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_pointcloud_;
    ros::Subscriber sub_odom_;
    ros::Publisher pub_pointcloud_static_;
    ros::Publisher pub_pointcloud_raw_;
    ros::Publisher pub_bbox_;
    ros::Publisher pub_dynamic_bbox_;
    ros::Publisher pub_pointcloud_cluster_;
    ros::Publisher pub_text_vel_;
    ros::Publisher pub_center_points_;

    cudaEvent_t start_, stop_;
    cudaStream_t stream_ = NULL;

    Params params;

    std::string Model_File_Dir_;
    std::string odom_frame_;
    std::string child_frame_;

    CloudTPtr original_scan_;

    double MINIMUM_RANGE;
    double MAXMUM_RANGE;
    bool crop_use_;
    double crop_size_;

    bool vf_use_;
    double vf_res_;

    pcl::CropBox<PointT> crop;
    pcl::VoxelGrid<pcl::PointXYZI> vf;
    std::deque<nav_msgs::Odometry> odom_queue;
    visualization_msgs::MarkerArray center_points_array;

    pcl::PointCloud<pcl::PointXYZI>::Ptr global_static_map_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr  global_dynamic_mask_;


    std::string save_dir_;
    bool save_on_shutdown_;

    bool verbose = true;

    std::unique_ptr<CenterPoint> center_pointpillars_ptr_;

    void PointCloud_Callback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void Odometry_Callback(const nav_msgs::OdometryPtr &odom);
    void publishCloud(std_msgs::Header header, const CloudTPtr& in_cloud_to_publish_ptr);
    void publishObjectBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> filter_BBox);
    void publishDynamicBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> dynamic_BBox);
    void publishClusterCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<pcl::PointIndices> cluster_indices);

    void preprocessPoints(const CloudTPtr& cloud_in, float th1, float th2);
    void removeClosedPointCloud(const CloudT &cloud_in, CloudT &cloud_out, float th1, float th2);
    void publishClusterRaw(std_msgs::Header header,const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);
    // --- 历史框并集相关（私有） ---
    bool nearestPose_(ros::Time t, Pose& out);
    bool relativeT_(ros::Time ta, ros::Time tb, Eigen::Matrix4f& T_ba);
    Bndbox transformBox_(const Bndbox& b, const Eigen::Matrix4f& T);
    void augmentWithHistory_(std::vector<Bndbox>& boxes_now, ros::Time t_now);

    // --- 串行队列与worker ---
    std::mutex mtx_cloud_;
    std::condition_variable cv_cloud_;
    std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_q_;
    size_t cloud_q_max_ = 3;       // 队列最多保留3帧，避免越积越大
    bool exit_worker_ = false;
    std::thread worker_;

    // --- odom历史互斥（odom回调与worker会并发访问） ---
    std::mutex mtx_odom_;

    sensor_msgs::PointCloud2ConstPtr latest_msg_;
    bool has_latest_ = false;

    void processingLoop_();
    bool poseAt_(ros::Time t, Eigen::Matrix4f& T_odom_child);   // 取/插值该时刻里程计
    void processOneCloud_(const sensor_msgs::PointCloud2ConstPtr& msg,
                          const Eigen::Matrix4f& T_odom_child, bool has_odom);

    float* d_points_ = nullptr;
    float* h_points_ = nullptr;  // pinned
    size_t max_points_ = MAX_POINTS_NUM;

    struct BoxParam {
        float cx, cy, cz;
        float c, s;     // cos(yaw), sin(yaw)
        float hx, hy, hz;
        float r2;       // (hx,hy) 外接圆半径^2，用于快速门限
    };

    // ---- timing stats ----
    struct Stat {
        double ema_ms = 0.0;
        double last_ms = 0.0;
        bool inited = false;
        void add(double ms, double alpha=0.1) {
            last_ms = ms;
            if (!inited) { ema_ms = ms; inited = true; }
            else ema_ms = (1.0-alpha)*ema_ms + alpha*ms;
        }
    };

    std::atomic<uint64_t> recv_cnt_{0};
    std::atomic<uint64_t> proc_cnt_{0};
    std::atomic<uint64_t> dropq_cnt_{0};

    std::atomic<uint64_t> bypass_cnt_{0};  // lag 太大走 bypass 分支的帧数
    std::atomic<uint64_t> empty_cnt_{0};   // boxes_to_remove.empty() 早退的帧数

    std::atomic<int>      q_size_{0};      // 当前 cloud_q_ 队列长度（用于打印）

    Stat t_fromros_, t_norm_, t_pack_, t_h2d_, t_infer_;
    Stat t_hist_, t_extract_, t_pre_, t_pub_, t_total_;

    ros::WallTime last_report_wall_ = ros::WallTime(0);
    ros::Time last_proc_stamp_ = ros::Time(0);
};

Center_PointPillars_ROS::Center_PointPillars_ROS(ros::NodeHandle nh) : nh_(nh) {
    const std::string pkg_trlo = ros::package::getPath("fast_livo");
    std::string model_dir   = pkg_trlo + "/model/center_pointpillars/";
    ros::param::param<std::string>("Model_File_Dir", this->Model_File_Dir_, model_dir);
    ros::param::param<std::string>("~center_pp/frame/odom_frame", this->odom_frame_, "robot/odom");
    ros::param::param<std::string>("~center_pp/frame/child_frame", this->child_frame_, "robot/base_link");

    ros::param::param<double>("~center_pp/preprocessing/threshold/MINIMUM_RANGE", this->MINIMUM_RANGE, 0.5);
    ros::param::param<double>("~center_pp/preprocessing/threshold/MAXMUM_RANGE", this->MAXMUM_RANGE, 80);

    // Crop Box Filter
    ros::param::param<bool>("~center_pp/preprocessing/cropBoxFilter/use", this->crop_use_, false);
    ros::param::param<double>("~center_pp/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

    // Voxel Grid Filter
    ros::param::param<bool>("~center_pp/preprocessing/voxelFilter/use", this->vf_use_, true);
    ros::param::param<double>("~center_pp/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

    ros::param::param<double>("~center_pp/filter_history/coast_time_sec", this->coast_time_sec_, 0.8);
    ros::param::param<int>   ("~center_pp/filter_history/coast_max_frames", this->coast_max_frames_, 8);
    ros::param::param<double>("~center_pp/filter_history/inflate_ratio", this->inflate_ratio_, 0.12);
    ros::param::param<double>("~center_pp/filter_history/inflate_m", this->inflate_m_, 0.25);
    ros::param::param<double>("~center_pp/filter_history/merge_center_thresh", this->merge_center_thresh_, 0.5);
    ros::param::param<bool>("~center_pp/normalize_world_cloud_to_local",
                            this->normalize_world_cloud_to_local_, true);
    ros::param::param<std::string>("~center_pp/save_dir", this->save_dir_, "/tmp");
    ros::param::param<bool>("~center_pp/save_on_shutdown", this->save_on_shutdown_, true);

    this->global_static_map_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    this->global_dynamic_mask_.reset(new pcl::PointCloud<pcl::PointXYZ>());



    checkCudaErrors(cudaEventCreate(&this->start_));
    checkCudaErrors(cudaEventCreate(&this->stop_));
    GPU_CHECK(cudaStreamCreate(&this->stream_));
    this->center_pointpillars_ptr_.reset(new CenterPoint(this->Model_File_Dir_, this->verbose)); // 外部定义调用不了cuda函数
    this->original_scan_.reset(new CloudT());
    this->crop.setNegative(true);
    this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
    this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

    center_pointpillars_ptr_->prepare();
    setlocale(LC_ALL,"");

    worker_ = std::thread(&Center_PointPillars_ROS::processingLoop_, this);

    cudaMalloc(&d_points_, max_points_ * 5 * sizeof(float));
    cudaHostAlloc(&h_points_, max_points_ * 5 * sizeof(float), cudaHostAllocDefault); // pinned
}

Center_PointPillars_ROS::~Center_PointPillars_ROS(){
    {
        std::lock_guard<std::mutex> lk(mtx_cloud_);
        exit_worker_ = true;
    }
    cv_cloud_.notify_all();
    if (worker_.joinable()) worker_.join();

    checkCudaErrors(cudaEventDestroy(this->start_));
    checkCudaErrors(cudaEventDestroy(this->stop_));
    checkCudaErrors(cudaStreamDestroy(this->stream_));

    if (d_points_) cudaFree(d_points_);
    if (h_points_) cudaFreeHost(h_points_);

    if (save_on_shutdown_) {
        // ---- 1. 直接保存全局静态地图：PointXYZI ----
        if (!global_static_map_->empty()) {
            global_static_map_->width  = global_static_map_->points.size();
            global_static_map_->height = 1;
            global_static_map_->is_dense = true;

            std::string static_path = save_dir_ + "/all_static_point_cloud_map.pcd";
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(static_path, *global_static_map_); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
//            PointCloudXYZI().swap(*global_static_map_);
//            // 二进制保存，体积更小；如果你想 ASCII，就换成 savePCDFileASCII
//            pcl::io::savePCDFileBinary(static_path, *global_static_map_);
            ROS_INFO_STREAM("[centerpp] Saved global static map to " << static_path);
        }

        // ---- 2. 直接保存全局动态 mask：PointXYZ ----
        if (!global_dynamic_mask_->empty()) {
            global_dynamic_mask_->width  = global_dynamic_mask_->points.size();
            global_dynamic_mask_->height = 1;
            global_dynamic_mask_->is_dense = true;

            std::string dyn_path = save_dir_ + "/all_dynamic_mask.pcd";
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(dyn_path , *global_dynamic_mask_); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
//            PointCloudXYZ().swap(*global_dynamic_mask_);
//            pcl::io::savePCDFileBinary(dyn_path, *global_dynamic_mask_);
            ROS_INFO_STREAM("[centerpp] Saved global dynamic mask to " << dyn_path);
        }
    }
}

void Center_PointPillars_ROS::Process() {
    std::cout << "Ready to receive point cloud topic!" << std::endl;
    this->sub_odom_ = nh_.subscribe ("odom", 160, &Center_PointPillars_ROS::Odometry_Callback, this, ros::TransportHints().tcpNoDelay());
    this->sub_pointcloud_ = nh_.subscribe("pointcloud", 1, &Center_PointPillars_ROS::PointCloud_Callback, this, ros::TransportHints().tcpNoDelay());
//    this->sub_odom_ = nh_.subscribe ("odom", 160, &Center_PointPillars_ROS::Odometry_Callback, this);
//    this->pub_pointcloud_static_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_static", 10);
//    this->pub_pointcloud_raw_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_raw", 10);
    this->pub_pointcloud_static_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_static", 1);
    this->pub_pointcloud_raw_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_raw", 1);
    this->pub_bbox_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("box", 10, true);
    this->pub_dynamic_bbox_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("dynamic_box", 10, true);
    this->pub_pointcloud_cluster_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_cluster", 1);
    //    this->pub_pointcloud_cluster_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_cluster", 10);
    this->pub_text_vel_ = nh_.advertise<visualization_msgs::MarkerArray>("centerpoint_vel", 10);
    this->pub_center_points_ = nh_.advertise<visualization_msgs::MarkerArray> ("center_markers", 10);
    ros::spin();
}

void Center_PointPillars_ROS::PointCloud_Callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    recv_cnt_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(mtx_cloud_);
    cloud_q_.push_back(msg);

    if (cloud_q_.size() > cloud_q_max_) {
        cloud_q_.pop_front();   // 丢最老
        dropq_cnt_.fetch_add(1, std::memory_order_relaxed);
    }

    q_size_.store((int)cloud_q_.size(), std::memory_order_relaxed);
    cv_cloud_.notify_one();
}

void Center_PointPillars_ROS::Odometry_Callback(const nav_msgs::OdometryPtr &odom)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf q(odom->pose.pose.orientation.w,
                         odom->pose.pose.orientation.x,
                         odom->pose.pose.orientation.y,
                         odom->pose.pose.orientation.z);
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T.block<3,1>(0,3) = Eigen::Vector3f(odom->pose.pose.position.x,
                                        odom->pose.pose.position.y,
                                        odom->pose.pose.position.z);

    {
        std::lock_guard<std::mutex> lk(mtx_odom_);
        odom_hist_.push_back({odom->header.stamp, T});
        while (!odom_hist_.empty() &&
               (odom_hist_.back().stamp - odom_hist_.front().stamp).toSec() > 5.0) {
            odom_hist_.pop_front();
        }
    }
}

void Center_PointPillars_ROS::processingLoop_()
{
    while (ros::ok()) {
        sensor_msgs::PointCloud2ConstPtr msg;
        {
            std::unique_lock<std::mutex> lk(mtx_cloud_);
            cv_cloud_.wait(lk, [&]{ return exit_worker_ || !cloud_q_.empty(); });
            if (exit_worker_) break;

            msg = cloud_q_.front();
            cloud_q_.pop_front();
            q_size_.store((int)cloud_q_.size(), std::memory_order_relaxed);
        }

        // “弹出=开始处理” 计数（不受 early return 影响）
        proc_cnt_.fetch_add(1, std::memory_order_relaxed);

        Eigen::Matrix4f T_odom_child = Eigen::Matrix4f::Identity();
        bool has_odom = poseAt_(msg->header.stamp, T_odom_child);
        processOneCloud_(msg, T_odom_child, has_odom);
    }
}

bool Center_PointPillars_ROS::poseAt_(ros::Time t, Eigen::Matrix4f& T_odom_child)
{
    std::lock_guard<std::mutex> lk(mtx_odom_);
    if (odom_hist_.empty()) return false;

    size_t best = 0;
    double bestdt = 1e9;
    for (size_t i = 0; i < odom_hist_.size(); ++i) {
        double dt = std::fabs((odom_hist_[i].stamp - t).toSec());
        if (dt < bestdt) { bestdt = dt; best = i; }
    }
    T_odom_child = odom_hist_[best].T;
    return true;
}

void Center_PointPillars_ROS::processOneCloud_(const sensor_msgs::PointCloud2ConstPtr& msg,
                                               const Eigen::Matrix4f& T_odom_child, bool has_odom)
{


    double T_total0 = wall_now_ms();

    // 如果你在用 /use_sim_time，这里 ros::Time::now() 是仿真时间，同样成立
    const double lag = (ros::Time::now() - msg->header.stamp).toSec();

    static double max_allow_lag = 0.20; // 建议参数化
    if (lag > max_allow_lag) {
        bypass_cnt_.fetch_add(1, std::memory_order_relaxed);

        CloudTPtr cloud(new CloudT());
        pcl::fromROSMsg(*msg, *cloud);

        this->preprocessPoints(cloud, this->MINIMUM_RANGE, this->MAXMUM_RANGE);
        this->publishCloud(msg->header, cloud);

        ROS_WARN_THROTTLE(1.0, "[centerpp] lag=%.3f > %.3f, bypass centerpoint for sync",
                          lag, max_allow_lag);
        return;
    }

    double t0 = wall_now_ms();
    CloudTPtr in_cloud_ptr(new CloudT());
    pcl::fromROSMsg(*msg, *in_cloud_ptr);
    t_fromros_.add(wall_now_ms() - t0);

    double tN0 = wall_now_ms();
    // ---- 如果输入是世界系（camera_init/map/world），先拉回到本地(child_frame_) ----
    auto in_frame = msg->header.frame_id;
    bool is_world_frame =
            (in_frame == "camera_init" || in_frame == "map" || in_frame == "world");

    if (this->normalize_world_cloud_to_local_ && is_world_frame) {
        // 取最接近当前点云时间戳的里程计（你已经有 nearestPose_）
        Pose P;
        if (this->nearestPose_(msg->header.stamp, P)) {
            // P.T = T_wc = 世界->child_frame_ 的位姿（来自 Odometry）
            const Eigen::Matrix4f& T_wc = P.T;

            // 求 child<-world 变换：T_cw = inv(T_wc)
            Eigen::Matrix4f T_cw = Eigen::Matrix4f::Identity();
            T_cw.block<3,3>(0,0) = T_wc.block<3,3>(0,0).transpose();
            T_cw.block<3,1>(0,3) = -T_cw.block<3,3>(0,0) * T_wc.block<3,1>(0,3);

            // 就地把点云从世界系拉回到 child_frame_ 局部系
            pcl::transformPointCloud(*in_cloud_ptr, *in_cloud_ptr, T_cw);

            // 让下游表意一致（可选）：标注为 child_frame_
            in_cloud_ptr->header.frame_id = this->child_frame_;
        } else {
            ROS_WARN_THROTTLE(1.0,
                              "[centerpp] no odom near stamp=%.3f, skip world->local normalization",
                              msg->header.stamp.toSec());
        }
    }

    t_norm_.add(wall_now_ms() - tN0);

    double tp0 = wall_now_ms();
    // 1) pack to pinned host (no disk)
    size_t points_num = packCloudToPillarsInput(
            *in_cloud_ptr, h_points_, max_points_,
            (float)MINIMUM_RANGE, (float)MAXMUM_RANGE);
    t_pack_.add(wall_now_ms() - tp0);

    double th0 = wall_now_ms();
    // 2) H2D async copy on your stream
    GPU_CHECK(cudaMemcpyAsync(
            d_points_, h_points_,
            points_num * 5 * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_));
    GPU_CHECK(cudaStreamSynchronize(stream_)); // 这里只为计时准确（更推荐用 cudaEvent）
    t_h2d_.add(wall_now_ms() - th0);

    double ti0 = wall_now_ms();
    // 3) inference (runs on stream)
    cudaEventRecord(start_, stream_);
    double t1 = ros::Time::now().toSec();
    center_pointpillars_ptr_->doinfer((void*)d_points_, points_num, stream_);
    GPU_CHECK(cudaStreamSynchronize(stream_));  // 等推理完成再读 nms_pred_
    double t2 = ros::Time::now().toSec();
    cudaEventRecord(stop_, stream_);

    t_infer_.add(wall_now_ms() - ti0);

    // timing（可选：建议不要每帧都同步 event）
    avg_centerpoint_time.push_back((t2 - t1) * 1000);


    std::vector<Bndbox> filter_BBox;
    for (auto box : this->center_pointpillars_ptr_->nms_pred_) {
        // car(id=0)/pedestrain(id=8)/cyclists(id=6)/truck(id=3)
        if (   (box.id == 0 && box.score > 0.5)    // car
               || (box.id == 6 && box.score > 0.75)   // cyclist
               || (box.id == 8 && box.score > 0.30)   // pedestrian
                )
        {
            filter_BBox.push_back(box);   // 都当作需要剔除的“动态目标”
        }
    }

    std::vector<Bndbox> dynamic_BBox = filter_BBox;
    std::vector<Bndbox> boxes_to_remove = filter_BBox;


    if (boxes_to_remove.empty()) {
        empty_cnt_.fetch_add(1, std::memory_order_relaxed);
        // 1) 预处理 + 发布：这里 in_cloud_ptr 还是 CloudTPtr（含time/ring），不会丢字段
        this->preprocessPoints(in_cloud_ptr, this->MINIMUM_RANGE, this->MAXMUM_RANGE);
        this->publishCloud(msg->header, in_cloud_ptr);

        this->publishObjectBoundingBox(msg->header, filter_BBox);
        this->publishDynamicBoundingBox(msg->header, dynamic_BBox);

        // 2) 累积全局地图：global_static_map_ 是 PointXYZI，所以要先把 CloudT 转成 XYZI
        pcl::PointCloud<pcl::PointXYZI> in_xyzi = toXYZI(*in_cloud_ptr);

        cpp::Center_PointPillars_ROS::Pose P_world;
        if (this->nearestPose_(msg->header.stamp, P_world)) {
            const Eigen::Matrix4f& T_odom_child = P_world.T;

            pcl::PointCloud<pcl::PointXYZI> in_world;
            pcl::transformPointCloud(in_xyzi, in_world, T_odom_child);
            *this->global_static_map_ += in_world;
        } else {
            *this->global_static_map_ += in_xyzi;
        }

        return;
    }


    CloudTPtr out_cloud_ptr(new CloudT());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>());

    double thist0 = wall_now_ms();
// 用历史补洞：把 boxes_to_remove 在过去若干帧的并集考虑进来
    std::vector<Bndbox> active = boxes_to_remove;
    if (has_odom) this->augmentWithHistory_(active, msg->header.stamp);
    t_hist_.add(wall_now_ms() - thist0);

    double tex0 = wall_now_ms();
// 从 active 这些区域里抠除点云
    this->extractBBoxPointcloud(active, in_cloud_ptr, out_cloud_ptr, cloud_cluster);
    t_extract_.add(wall_now_ms() - tex0);

    double avg_centerpoint_totaltime = std::accumulate(avg_centerpoint_time.begin(), avg_centerpoint_time.end(), 0.0) / avg_centerpoint_time.size();
    std::cout << "CenterPoint Time :: " << std::setfill(' ') << std::setw(6) << avg_centerpoint_time.back() << " ms    // Avg: " << std::setw(5) << avg_centerpoint_totaltime << std::endl;

    double tpre0 = wall_now_ms();
    this->preprocessPoints(out_cloud_ptr, this->MINIMUM_RANGE, this->MAXMUM_RANGE);
    t_pre_.add(wall_now_ms() - tpre0);

    double tpub0 = wall_now_ms();
    this->publishCloud(msg->header, out_cloud_ptr);
    this->publishObjectBoundingBox(msg->header, filter_BBox);
    this->publishDynamicBoundingBox(msg->header, dynamic_BBox);
//    this->publishClusterCloud(msg->header, cloud_cluster, cluster_indices);
    cloud_cluster->width  = cloud_cluster->points.size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    this->publishClusterRaw(msg->header, cloud_cluster);
    t_pub_.add(wall_now_ms() - tpub0);
    // 记录本帧检测到的原始框（不膨胀）
    this->box_hist_.push_back({msg->header.stamp, filter_BBox});
    // 历史上限（例：2 秒）
    while (!this->box_hist_.empty() &&
           (msg->header.stamp - this->box_hist_.front().stamp).toSec() > 2.0) {
        this->box_hist_.pop_front();
    }

    // ---- 利用最近的里程计位姿，把当前帧点云变到世界系(odom_frame_)后再叠加 ----
    cpp::Center_PointPillars_ROS::Pose P_world;

    // 先把 CloudT(velodyne_ros::Point) 转成 PointXYZI
    pcl::PointCloud<pcl::PointXYZI> out_xyzi = toXYZI(*out_cloud_ptr);

    if (this->nearestPose_(msg->header.stamp, P_world)) {
        const Eigen::Matrix4f& T_odom_child = P_world.T;  // child_frame_ -> odom_frame_

        // 静态点云：XYZI
        pcl::PointCloud<pcl::PointXYZI> out_world;
        pcl::transformPointCloud(out_xyzi, out_world, T_odom_child);
        *this->global_static_map_ += out_world;

        // 动态点 mask：XYZ
        pcl::PointCloud<pcl::PointXYZ> cluster_world;
        pcl::transformPointCloud(*cloud_cluster, cluster_world, T_odom_child);
        *this->global_dynamic_mask_ += cluster_world;
    } else {
        ROS_WARN_THROTTLE(1.0,
                          "[centerpp] no odom for stamp %.3f, accumulate in local frame",
                          msg->header.stamp.toSec());

        // 找不到里程计：直接叠加本地系的 XYZI（注意不是 out_cloud_ptr）
        *this->global_static_map_ += out_xyzi;
        *this->global_dynamic_mask_ += *cloud_cluster;
    }

    t_total_.add(wall_now_ms() - T_total0);

    ros::WallTime noww = ros::WallTime::now();
    if ((noww - last_report_wall_).toSec() > 1.0) {
        last_report_wall_ = noww;

        uint64_t recv   = recv_cnt_.load(std::memory_order_relaxed);
        uint64_t proc   = proc_cnt_.load(std::memory_order_relaxed);
        uint64_t dropQ  = dropq_cnt_.load(std::memory_order_relaxed);
        uint64_t byp    = bypass_cnt_.load(std::memory_order_relaxed);
        uint64_t emp    = empty_cnt_.load(std::memory_order_relaxed);
        int qsz         = q_size_.load(std::memory_order_relaxed);

        long diff = (long)(recv - proc);

        double stamp_gap = 0.0;
        if (!last_proc_stamp_.isZero()) {
            stamp_gap = (msg->header.stamp - last_proc_stamp_).toSec();
        }
        last_proc_stamp_ = msg->header.stamp;

        ROS_WARN("[centerpp][timing][EMA ms] fromROS=%.2f norm=%.2f pack=%.2f h2d=%.2f infer=%.2f hist=%.2f extract=%.2f pre=%.2f pub=%.2f TOTAL=%.2f | recv=%lu proc=%lu dropQ=%lu diff=%ld q=%d | bypass=%lu empty=%lu | stamp_gap=%.3f",
                 t_fromros_.ema_ms, t_norm_.ema_ms, t_pack_.ema_ms, t_h2d_.ema_ms, t_infer_.ema_ms,
                 t_hist_.ema_ms, t_extract_.ema_ms, t_pre_.ema_ms, t_pub_.ema_ms,
                 t_total_.ema_ms,
                 (unsigned long)recv, (unsigned long)proc, (unsigned long)dropQ, diff, qsz,
                 (unsigned long)byp, (unsigned long)emp,
                 stamp_gap);
    }
}



void Center_PointPillars_ROS::extractBBoxPointcloud(
        std::vector<Bndbox> filter_BBox,
        const CloudTPtr& cloud_in,
        CloudTPtr& cloud_out,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cluster)
{
    if (filter_BBox.empty()) { *cloud_out = *cloud_in; return; }

    const bool  YAW_FLIP = true;
    const float SCALE_XY = 1.2f;
    const float PAD_X = 0.35f, PAD_Y = 0.35f, PAD_Z = 0.40f;

    std::vector<BoxParam> B; B.reserve(filter_BBox.size());
    for (const auto& b : filter_BBox) {
        float yaw = (float)b.rt;
        if (YAW_FLIP) { yaw = -yaw - (float)M_PI/2.0f; }
        float c = std::cos(yaw), s = std::sin(yaw);

        float hx = 0.5f*(float)b.l*SCALE_XY + PAD_X;
        float hy = 0.5f*(float)b.w*SCALE_XY + PAD_Y;
        float hz = 0.5f*(float)b.h*1.05f    + PAD_Z;
        float r2 = (hx*hx + hy*hy); // 外接圆半径^2（保守一点也行）

        B.push_back({(float)b.x,(float)b.y,(float)b.z, c,s, hx,hy,hz, r2});
    }

    const int N = (int)cloud_in->size();
    std::vector<uint8_t> rm(N, 0);

    // 可选：OpenMP 并行（有的话速度提升很明显）
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const auto& pi = cloud_in->points[i];
        const float px = pi.x, py = pi.y, pz = pi.z;

        for (const auto& bp : B) {
            float dx = px - bp.cx, dy = py - bp.cy;
            if (dx*dx + dy*dy > bp.r2) continue;  // 快速圆门限

            float dz = pz - bp.cz;
            if (std::fabs(dz) > bp.hz) continue;

            float lx =  dx*bp.c + dy*bp.s;
            float ly = -dx*bp.s + dy*bp.c;
            if (std::fabs(lx) <= bp.hx && std::fabs(ly) <= bp.hy) {
                rm[i] = 1;
                break;
            }
        }
    }

    cloud_out->points.clear();
    cloud_cluster->points.clear();
    cloud_out->points.reserve(cloud_in->size());
    cloud_cluster->points.reserve(std::min<size_t>(cloud_in->size(), 100000));

    for (int i = 0; i < N; ++i) {
        const auto& pi = cloud_in->points[i];
        if (rm[i]) cloud_cluster->points.push_back(pcl::PointXYZ{pi.x,pi.y,pi.z});
        else       cloud_out->points.push_back(pi); // 保留 time/ring
    }

    cloud_out->width  = (uint32_t)cloud_out->points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = true;
}



// 取时间 t 最近的里程计
bool Center_PointPillars_ROS::nearestPose_(ros::Time t, Pose& out) {
    if (odom_hist_.empty()) return false;
    size_t best = 0; double bestdt = 1e9;
    for (size_t i = 0; i < odom_hist_.size(); ++i) {
        double dt = std::fabs((odom_hist_[i].stamp - t).toSec());
        if (dt < bestdt) { bestdt = dt; best = i; }
    }
    out = odom_hist_[best];
    return true;
}

// 计算 T_ba = T_b * inv(T_a)
bool Center_PointPillars_ROS::relativeT_(ros::Time ta, ros::Time tb, Eigen::Matrix4f& T_ba) {
    Pose A, B;
    if (!nearestPose_(ta, A) || !nearestPose_(tb, B)) return false;
    Eigen::Matrix4f Ainv = Eigen::Matrix4f::Identity();
    Ainv.block<3,3>(0,0) = A.T.block<3,3>(0,0).transpose();
    Ainv.block<3,1>(0,3) = -Ainv.block<3,3>(0,0) * A.T.block<3,1>(0,3);
    T_ba = B.T * Ainv;
    return true;
}

// 把框按 4x4 变换到当前时刻，并做尺寸膨胀
Bndbox Center_PointPillars_ROS::transformBox_(const Bndbox& b, const Eigen::Matrix4f& T) {
    Eigen::Vector4f c(b.x, b.y, b.z, 1.f);
    Eigen::Vector4f c_new = T * c;
    // XY 平面旋转近似
    float yaw_delta = std::atan2(T(1,0), T(0,0));

    Bndbox o = b;
    o.x = c_new.x(); o.y = c_new.y(); o.z = c_new.z();
    o.rt = b.rt + yaw_delta;
    o.l  = b.l * (1.0 + inflate_ratio_) + 2.0 * inflate_m_;
    o.w  = b.w * (1.0 + inflate_ratio_) + 2.0 * inflate_m_;
    o.h  = b.h * (1.0 + inflate_ratio_) + 2.0 * inflate_m_;
    return o;
}

// 把历史框并入当帧
void Center_PointPillars_ROS::augmentWithHistory_(std::vector<Bndbox>& boxes_now, ros::Time t_now) {
    int used = 0;
    for (auto it = box_hist_.rbegin(); it != box_hist_.rend(); ++it) {
        if ((t_now - it->stamp).toSec() > coast_time_sec_) break;
        if (used >= coast_max_frames_) break;

        Eigen::Matrix4f T; // T_now_prev: prev -> now
        if (!relativeT_(it->stamp, t_now, T)) { ++used; continue; }

        for (const auto& b : it->boxes) {
            Bndbox bb = transformBox_(b, T);

            // 与当前已有框做一个简单合并（中心距离）
            bool dup = false;
            for (const auto& cur : boxes_now) {
                float dx = bb.x - cur.x, dy = bb.y - cur.y;
                if (dx*dx + dy*dy < static_cast<float>(merge_center_thresh_ * merge_center_thresh_)) {
                    dup = true; break;
                }
            }
            if (!dup) boxes_now.push_back(bb);
        }
        ++used;
    }
}

void Center_PointPillars_ROS::publishCloud(std_msgs::Header header,
                                           const CloudTPtr& in_cloud_to_publish_ptr)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);

    static bool printed=false;
    if(!printed){
        std::ostringstream ss;
        ss << "[centerpp] OUT fields: ";
        for(const auto& f: cloud_msg.fields) ss << f.name << " ";
        ROS_WARN_STREAM(ss.str());
        printed=true;
    }

    cloud_msg.header = header;
    cloud_msg.header.frame_id = this->child_frame_;
    this->pub_pointcloud_static_.publish(cloud_msg);
}



void Center_PointPillars_ROS::publishObjectBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> filter_BBox) {

    jsk_recognition_msgs::BoundingBoxArray arr_bbox;
    int i = 0;

    for (const auto box : filter_BBox) {
        jsk_recognition_msgs::BoundingBox bbox;

        bbox.header = in_msg_header;
        bbox.header.frame_id = this->child_frame_;
        bbox.pose.position.x =  box.x;
        bbox.pose.position.y =  box.y;
        bbox.pose.position.z = box.z;
        bbox.dimensions.x = box.w;  // width
        bbox.dimensions.y = box.l;  // length
        bbox.dimensions.z = box.h;  // height
        // Using tf::Quaternion for quaternion from roll, pitch, yaw
        tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, -box.rt);
        bbox.pose.orientation.x = q.x();
        bbox.pose.orientation.y = q.y();
        bbox.pose.orientation.z = q.z();
        bbox.pose.orientation.w = q.w();
        bbox.value = box.score;
        bbox.label = box.id;
        arr_bbox.boxes.push_back(bbox);
        // if(box.score>0.5){
        // arr_bbox.boxes.push_back(bbox);
        // }

    }
    // std::cout<<"find bbox Num:"<<arr_bbox.boxes.size()<<std::endl;
    arr_bbox.header = in_msg_header;
    arr_bbox.header.frame_id = this->child_frame_;

    this->pub_bbox_.publish(arr_bbox);
}


void Center_PointPillars_ROS::publishDynamicBoundingBox(std_msgs::Header in_msg_header, std::vector<Bndbox> dynamic_BBox) {

    this->center_points_array.markers.clear();
    this->center_points_array.markers.reserve(dynamic_BBox.size());

    jsk_recognition_msgs::BoundingBoxArray arr_bbox;
    visualization_msgs::MarkerArray text_vel_array;
    visualization_msgs::Marker text_vel, center_points;
    int id = 0;
    center_points.lifetime = ros::Duration();
    center_points.header = in_msg_header;
    center_points.header.frame_id = this->child_frame_;
    center_points.ns = "center_points";
    center_points.action = visualization_msgs::Marker::ADD;
    center_points.type = visualization_msgs::Marker::POINTS;
    center_points.scale.x = 0.7;
    center_points.scale.y = 0.7;
    center_points.scale.z = 0.7;

    for (const auto box : dynamic_BBox) {
        jsk_recognition_msgs::BoundingBox bbox;

        bbox.header = in_msg_header;
        bbox.header.frame_id = this->child_frame_;  // Replace with your frame_id
        bbox.pose.position.x =  box.x;
        bbox.pose.position.y =  box.y;
        bbox.pose.position.z = box.z;
        bbox.dimensions.x = box.w;  // width
        bbox.dimensions.y = box.l;  // length
        bbox.dimensions.z = box.h;  // height
        // Using tf::Quaternion for quaternion from roll, pitch, yaw
        tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, -box.rt);
        bbox.pose.orientation.x = q.x();
        bbox.pose.orientation.y = q.y();
        bbox.pose.orientation.z = q.z();
        bbox.pose.orientation.w = q.w();
        bbox.value = box.score;
        bbox.label = box.id;
        arr_bbox.boxes.push_back(bbox);
        // if(box.score>0.5){
        // arr_bbox.boxes.push_back(bbox);
        // }

        text_vel.header = in_msg_header;
        text_vel.header.frame_id = this->child_frame_;
        text_vel.ns = "dynamic_vel";
        text_vel.action = visualization_msgs::Marker::ADD;
        text_vel.id = id;
        text_vel.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_vel.scale.z = 1;
        text_vel.color.r = text_vel.color.b = text_vel.color.g =  1;
        text_vel.color.a = 1;
        float vel = box.vy;
        std::ostringstream oss;
        oss << std::setprecision(2) << vel;
        text_vel.text = oss.str() + "m/s";
        text_vel.pose.orientation.x = q.x();
        text_vel.pose.orientation.y = q.y();
        text_vel.pose.orientation.z = q.z();
        text_vel.pose.orientation.w = q.w();
        text_vel.pose.position.x = box.x;
        text_vel.pose.position.y = box.y;
        text_vel.pose.position.z = box.z + box.h / 2 + 0.1;
        text_vel_array.markers.push_back(text_vel);

        center_points.id = id;
        center_points.type = visualization_msgs::Marker::POINTS;
        int ci = (id % 100) * 3;
        center_points.color.r = color[ci];
        center_points.color.g = color[ci + 1];
        center_points.color.b = color[ci + 2];
        center_points.color.a = 0.7;
        center_points.pose.orientation.w = 1;
        geometry_msgs::Point p;
        p.x = box.x;
        p.y = box.y;
        p.z = box.z;
        center_points.points.push_back(p);

        this->center_points_array.markers.push_back(center_points);

        ++id;
    }
    // std::cout<<"find bbox Num:"<<arr_bbox.boxes.size()<<std::endl;
    arr_bbox.header = in_msg_header;
    arr_bbox.header.frame_id = this->child_frame_;

    this->pub_dynamic_bbox_.publish(arr_bbox);
    this->pub_text_vel_.publish(text_vel_array);
    this->pub_center_points_.publish(this->center_points_array);
}



void Center_PointPillars_ROS::publishClusterCloud(std_msgs::Header header, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<pcl::PointIndices> cluster_indices) {
    int color_index = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_point(new pcl::PointCloud<pcl::PointXYZRGB>());
    int clusterSize = cluster_indices.size();
    for (int i = 0; i < clusterSize; i++) {
        int clusterindixSize = cluster_indices[i].indices.size();
        for (int j = 0; j < clusterindixSize; j++) {
            pcl::PointXYZRGB point;
            point.x = cloud_in->points[cluster_indices[i].indices[j]].x;
            point.y = cloud_in->points[cluster_indices[i].indices[j]].y;
            point.z = cloud_in->points[cluster_indices[i].indices[j]].z;
            point.r = color[int(3) * color_index];
            point.g = color[int(3) * color_index + 1];
            point.b = color[int(3) * color_index + 2];
            color_point->push_back(point);
        }
        color_index++;
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*color_point, cloud_msg);
    cloud_msg.header = header;
    cloud_msg.header.frame_id = this->child_frame_;
    this->pub_pointcloud_cluster_.publish(cloud_msg);
}

void Center_PointPillars_ROS::publishClusterRaw(std_msgs::Header header,
                                                const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud_in, msg);
    msg.header = header;
    msg.header.frame_id = this->child_frame_;   // 和其它可视化一致
    this->pub_pointcloud_cluster_.publish(msg);
}

static inline void voxelDownsampleKeepFirst_(CloudT& cloud, float leaf)
{
    if (cloud.empty() || leaf <= 1e-6f) return;

    struct Key { int ix, iy, iz; };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = 1469598103934665603ULL;
            auto mix = [&](int v){
                h ^= (size_t)v + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            };
            mix(k.ix); mix(k.iy); mix(k.iz);
            return h;
        }
    };
    struct KeyEq {
        bool operator()(const Key& a, const Key& b) const noexcept {
            return a.ix==b.ix && a.iy==b.iy && a.iz==b.iz;
        }
    };

    std::unordered_map<Key, int, KeyHash, KeyEq> seen;
    seen.reserve(cloud.size()/4);

    CloudT out;
    out.header = cloud.header;
    out.is_dense = cloud.is_dense;
    out.points.reserve(cloud.points.size()/4);

    for (const auto& p : cloud.points) {
        Key k{
                (int)std::floor(p.x / leaf),
                (int)std::floor(p.y / leaf),
                (int)std::floor(p.z / leaf)
        };
        if (seen.emplace(k, 1).second) {
            out.points.push_back(p); // 原样保留 time/ring
        }
    }
    out.width  = (uint32_t)out.points.size();
    out.height = 1;
    cloud.swap(out);
}

void Center_PointPillars_ROS::preprocessPoints(const CloudTPtr& cloud_in, float th1, float th2)
{
    // raw 发布（可选）
    *this->original_scan_ = *cloud_in;
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*this->original_scan_, cloud_msg);
    cloud_msg.header.frame_id = this->child_frame_;
    this->pub_pointcloud_raw_.publish(cloud_msg);

    // Remove NaNs
    std::vector<int> idx;
    cloud_in->is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, idx);

    // Range filter（保留字段）
    this->removeClosedPointCloud(*cloud_in, *cloud_in, th1, th2);

    // Crop（不会改字段）
    if (this->crop_use_) {
        this->crop.setInputCloud(cloud_in);
        this->crop.filter(*cloud_in);
    }

    // Voxel（自定义：保留 ring/time）
    if (this->vf_use_) {
        voxelDownsampleKeepFirst_(*cloud_in, (float)this->vf_res_);
    }
}

void Center_PointPillars_ROS::removeClosedPointCloud(const CloudT &cloud_in,
                                                     CloudT &cloud_out,
                                                     float th1, float th2)
{
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        const float x = cloud_in.points[i].x;
        const float y = cloud_in.points[i].y;
        const float z = cloud_in.points[i].z;
        float dis = x*x + y*y + z*z;
        if (dis < th1 * th1) continue;
        if (dis > th2 * th2) continue;
        cloud_out.points[j++] = cloud_in.points[i]; // 整点复制：time/ring 保留
    }

    cloud_out.points.resize(j);
    cloud_out.height = 1;
    cloud_out.width  = (uint32_t)j;
    cloud_out.is_dense = true;
}




} // namespace cpp

int main(int argc, char **argv) {
    ros::init(argc, argv, "centerpp_node");
    ros::NodeHandle nh("~");

    color.clear();
    for (size_t i_segment = 0; i_segment < 100; i_segment++)
    {
        color.push_back(static_cast<unsigned char>(rand() % 256));
        color.push_back(static_cast<unsigned char>(rand() % 256));
        color.push_back(static_cast<unsigned char>(rand() % 256));
    }

    // GetDeviceInfo();
    initDevice(0);

    cpp::Center_PointPillars_ROS center_pintPillars_ros(nh);
    center_pintPillars_ros.Process();

    return 0;
}
