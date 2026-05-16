/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"
#include <unordered_set>
using Mat6 = Eigen::Matrix<double,6,6>;
using Vec6 = Eigen::Matrix<double,6,1>;

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config)
{
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);
  
  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_, 0.01);
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);
  nh.param<vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_, vector<int>{5,5,5,5,5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);
  nh.param<bool>("lio/Degeneracy_mitigation", voxel_config.Degeneracy_mitigation, false);
  nh.param<double>("lio/Degeneracy_Trans_thres", voxel_config.Degeneracy_Trans_thres, 10);
  nh.param<double>("lio/Degeneracy_Rot_thres", voxel_config.Degeneracy_Rot_thres, 10);

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en, false);
  nh.param<int>("local_map/half_map_size", voxel_config.half_map_size, 100);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh, 8);
  nh.param<double>("local_map/CAPACITY", voxel_config.CAPACITY, 50000);
  nh.param<double>("local_map/CAPACITY_FEATS_MAP", voxel_config.CAPACITY_FEATS_MAP, 2000);
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::init_octo_tree()
{
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cut_octo_tree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree()
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          init_plane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

// 轴映射：我们约定增量排列为 [roll pitch yaw x y z]。
// 若你的顺序是 [x y z roll pitch yaw]，请把两个 block 的索引对调。
// 轴级退化 → 按能量占比选择要抑制的本征向量列 → 抑制 dx 前 6 维，并输出每轴 0/1
void VoxelMapManager::suppressByEigMask(
        const Eigen::Matrix<double,6,6>& HTHdeal,
        const DegeneracyResult& degen,
        Eigen::Ref<Eigen::Matrix<double,6,1>> dx,
        double t)
{
    if (!degen.any) return;                   // 无退化，直接返回

    // ---- 1) 对称化 + 轻阻尼 ----
    Eigen::Matrix<double,6,6> H = 0.5*(HTHdeal + HTHdeal.transpose());
    H.diagonal().array() += 1e-9;

    // ---- 2) 特征分解 ----
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(H);
    if (es.info()!=Eigen::Success) return;
    const Eigen::Matrix<double,6,6>& U = es.eigenvectors();    // 正交阵，列升序

    // ---- 3) 依据“轴退化标志”选择要抑制的列（能量占比法）----
//    constexpr double kAlignThr = 0.5;  // 列与被标记轴的能量重合度阈值(0~1)
    std::array<int,6> keep{};          // 1=保留, 0=抑制
    keep.fill(1);
//
    for (int k=0; k<3; ++k) {
        if (degen.rot_degen[k])   keep[k] = 0;

        if (degen.trans_degen[k]) keep[k+3] = 0;
    }

    // 构造 D = diag(keep)
    Eigen::Matrix<double,6,6> D = Eigen::Matrix<double,6,6>::Zero();
    for (int k=0;k<6;++k) D(k,k) = static_cast<double>(keep[k]);

    // ---- 4) 在本征空间抑制增量：dx ← U * D * U^T * dx ----
    dx = U.inverse() * (U * D) * dx;

    // ---- 5) 由被抑制列得到每轴 0/1 并写日志 ----
    // Pbad = U * (I-D) * U^T，diag(Pbad) 反映各轴在坏子空间中的能量占比
//    Eigen::Matrix<double,6,6> Pbad =
//            U * (Eigen::Matrix<double,6,6>::Identity() - D) * U.transpose();

//    constexpr double kAxisThr = 0.5;  // 轴级阈值（>0.5 认为该轴大多落在坏子空间 → 置 1）
//    int f_roll  = (Pbad(0,0) > kAxisThr) ? 1 : 0;
//    int f_pitch = (Pbad(1,1) > kAxisThr) ? 1 : 0;
//    int f_yaw   = (Pbad(2,2) > kAxisThr) ? 1 : 0;
//    int f_x     = (Pbad(3,3) > kAxisThr) ? 1 : 0;
//    int f_y     = (Pbad(4,4) > kAxisThr) ? 1 : 0;
//    int f_z     = (Pbad(5,5) > kAxisThr) ? 1 : 0;
}

VoxelMapManager::AxisEig VoxelMapManager::align_eigs_to_axes(const Eigen::Vector3d& lam,
                           const Eigen::Matrix3d& U,
                           const Eigen::Vector3d& ex,
                           const Eigen::Vector3d& ey,
                           const Eigen::Vector3d& ez)
{
    // 对齐度矩阵 A(i,j) = |e_i^T u_j|
    Eigen::Matrix3d A;
    A(0,0)=std::abs(ex.dot(U.col(0))); A(0,1)=std::abs(ex.dot(U.col(1))); A(0,2)=std::abs(ex.dot(U.col(2)));
    A(1,0)=std::abs(ey.dot(U.col(0))); A(1,1)=std::abs(ey.dot(U.col(1))); A(1,2)=std::abs(ey.dot(U.col(2)));
    A(2,0)=std::abs(ez.dot(U.col(0))); A(2,1)=std::abs(ez.dot(U.col(1))); A(2,2)=std::abs(ez.dot(U.col(2)));

    // 枚举 6 个排列，选总对齐度最大的
    int best_p[3] = {0,1,2};
    double best_score = -1.0;
    int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for (auto& P : perms) {
        double s = A(0,P[0]) + A(1,P[1]) + A(2,P[2]);
        if (s > best_score) { best_score = s; best_p[0]=P[0]; best_p[1]=P[1]; best_p[2]=P[2]; }
    }

    AxisEig out;
    out.lam_axis << lam(best_p[0]), lam(best_p[1]), lam(best_p[2]);     // λ_x, λ_y, λ_z
    out.U_axis.col(0) = U.col(best_p[0]);                               // u_x
    out.U_axis.col(1) = U.col(best_p[1]);                               // u_y
    out.U_axis.col(2) = U.col(best_p[2]);                               // u_z
    return out;
}

void VoxelMapManager::Degeneracy_detection(const MD(6,6)& H_T_H,
    double t,
    double kappa_thr_trans,
    double kappa_thr_rot,
    DegeneracyResult& out)   // 可观测子空间（好）
{
    // 对称化 + 轻微阻尼，避免数值不正定
    MD(6,6) H = 0.5*(H_T_H + H_T_H.transpose());
    H += 1e-9 * MD(6,6)::Identity();

    // 旋转(0:2)、平移(3:5) —— 若你的状态顺序是 [x y z roll pitch yaw] 就把两行对调
    Eigen::Matrix3d H_rot   = H.block<3,3>(0,0);
    Eigen::Matrix3d H_trans = H.block<3,3>(3,3);

    // 自伴特征分解（升序：lam(0) <= lam(1) <= lam(2)）
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> esT(H_trans), esR(H_rot);
    const Eigen::Vector3d lamT = esT.eigenvalues();
    const Eigen::Vector3d lamR = esR.eigenvalues();
    const Eigen::Matrix3d UT   = esT.eigenvectors();   // 升序
    const Eigen::Matrix3d UR   = esR.eigenvectors();

    const Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();

    AxisEig Taxis = align_eigs_to_axes(lamT, UT, ex, ey, ez);   // 得到 λx,λy,λz（trans）
    AxisEig Raxis = align_eigs_to_axes(lamR, UR, ex, ey, ez);   // 得到 λroll,λpitch,λyaw（若旋转误差在该系）

    auto safe_min = [](double x){ return std::max(x, 1e-12); };
    out.kappa_trans = lamT(2) / safe_min(lamT(0));
    out.kappa_rot   = lamR(2) / safe_min(lamR(0));

    // 将“条件数阈值”换算为“特征值下界”：lam_min < lam_max / tau 视为退化
    const double lam_thr_T = lamT(2) / std::max(kappa_thr_trans, 1.0);
    const double lam_thr_R = lamR(2) / std::max(kappa_thr_rot  , 1.0);

    // 标记每个主方向是否退化（注意：这是3个“主方向”（特征向量），
    // 不一定与 x/y/z 或 roll/pitch/yaw 轴对齐）
    // 现在必须对齐！
    for (int i=0;i<3;++i) {
        out.trans_degen[i] = (Taxis.lam_axis(i) < lam_thr_T);
        out.rot_degen[i]   = (Raxis.lam_axis(i) < lam_thr_R);
    }

    out.any = (out.kappa_trans > kappa_thr_trans) || (out.kappa_rot > kappa_thr_rot);

    //max eigen value and ratio and inflate
    auto eps = 1e-12;
    out.lamT_max = lamT(2);
    out.lamR_max = lamR(2);

    out.lam_trans_axis = Taxis.lam_axis;   // [λx λy λz]
    out.lam_rot_axis   = Raxis.lam_axis;   // [λroll λpitch λyaw]

    // ratio = λmax / λaxis
    for(int i=0;i<3;++i){
        out.ratio_trans(i) = out.lamT_max / std::max(out.lam_trans_axis(i), eps);
        out.ratio_rot(i)   = out.lamR_max / std::max(out.lam_rot_axis(i), eps);

        // inflate = ratio^2 （对应你图 (32) 的形式）
        out.inflate_trans(i) = out.ratio_trans(i) * out.ratio_trans(i);
        out.inflate_rot(i)   = out.ratio_rot(i)   * out.ratio_rot(i);

        double cap = 1e6; // 你可调
        out.inflate_trans(i) = std::min(out.inflate_trans(i), cap);
        out.inflate_rot(i)   = std::min(out.inflate_rot(i),   cap);
    }

    // 追加写文件（若你已有成员 ofstream，可替换为成员变量）
    static std::ofstream fout_cond(std::string(ROOT_DIR) + "Log/degeneracy.txt",
                                   std::ios::out | std::ios::trunc);
    static std::ofstream fout_eigs(std::string(ROOT_DIR) + "Log/degeneracy_eigs.txt",
                                   std::ios::out | std::ios::trunc);

    // 时间戳 + 条件数
    fout_cond << std::fixed << std::setprecision(6) << t << " "
    << std::scientific << std::setprecision(6)
    << out.kappa_trans << " " << out.kappa_rot << "\n";

    // 时间戳 + 各轴特征值（平移3个 + 旋转3个）
    fout_eigs << std::fixed << std::setprecision(6) << t << " "
    << std::scientific << std::setprecision(6)
    << Taxis.lam_axis(0) << " " << Taxis.lam_axis(1) << " " << Taxis.lam_axis(2) << " "
    << Raxis.lam_axis(0) << " " << Raxis.lam_axis(1) << " " << Raxis.lam_axis(2) << "\n";

    // 注：如你希望“轴标志=探测标志”而非“投影判定”，可直接
    int f_roll = out.rot_degen[0];
    int f_pitch = out.rot_degen[1];
    int f_yaw = out.rot_degen[2];
    int f_x = out.trans_degen[0];
    int f_y = out.trans_degen[1];
    int f_z = out.trans_degen[2];

    static bool header_written = false;
    static std::ofstream flog(std::string(ROOT_DIR)+"Log/degeneracy_mask_axes.txt",
                              std::ios::out | std::ios::trunc);
    if (!header_written) {
        flog << "# time roll pitch yaw x y z    (1=degenerate/suppressed, 0=normal)\n";
        header_written = true;
    }
    flog << std::fixed << std::setprecision(6) << t << " "
        << f_roll  << " " << f_pitch << " " << f_yaw << " "
        << f_x     << " " << f_y     << " " << f_z   << "\n";
    flog.flush();
}


void VoxelMapManager::StateEstimation(StatesGroup &state_propagat, double last_lio_update_time, VoxelMapManager::DegeneracyResult& degen)
{
  cross_mat_list_.clear();
  cross_mat_list_.reserve(feats_down_size_);
  body_cov_list_.clear();
  body_cov_list_.reserve(feats_down_size_);

  // build_residual_time = 0.0;
  // ekf_time = 0.0;
  // double t0 = omp_get_wtime();

  for (size_t i = 0; i < feats_down_body_->size(); i++)
  {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    body_cov_list_.push_back(var);
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_.push_back(point_crossmat);
  }

  vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();

  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
  bool degen_done_this_frame_ = false;
  //DegeneracyResult degen;
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    double total_residual = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    M3D rot_var = state_.cov.block<3, 3>(0, 0);
    M3D t_var = state_.cov.block<3, 3>(3, 3);
    for (size_t i = 0; i < feats_down_body_->size(); i++)
    {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

      M3D cov = body_cov_list_[i];
      M3D point_crossmat = cross_mat_list_[i];
      cov = state_.rot_end * cov * state_.rot_end.transpose() + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = body_cov_list_[i];
    }
    ptpl_list_.clear();

    // double t1 = omp_get_wtime();

    BuildResidualListOMP(pv_list_, ptpl_list_);

    // build_residual_time += omp_get_wtime() - t1;

    for (int i = 0; i < ptpl_list_.size(); i++)
    {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size() << ", downsampled feature num:" << feats_down_size_ 
         << " effective feature num: " << effct_feat_num_ << " average residual: " << total_residual / effct_feat_num_ << endl;

    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
     * ***/
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
    VectorXd R_inv(effct_feat_num_);
    VectorXd meas_vec(effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_;
      V3D point_body(ptpl.point_b_);
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);

      /*** get the normal vector of closest surface/corner ***/

      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;

      M3D var;
      // V3D normal_b = state_.rot_end.inverse() * ptpl_list_[i].normal_;
      // V3D point_b = ptpl_list_[i].point_b_;
      // double cos_theta = fabs(normal_b.dot(point_b) / point_b.norm());
      // ptpl_list_[i].body_cov_ = ptpl_list_[i].body_cov_ * (1.0 / cos_theta) * (1.0 / cos_theta);

      // point_w cov
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) + (-point_crossmat) * state_propagat.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose();

      // point_w cov (another_version)
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) - point_crossmat * state_propagat.cov.block<3, 3>(0, 0) * point_crossmat;

      // point_body cov
      var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();

      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();

      R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
      // R_inv(i) = 1.0 / (sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);

      /*** calculate the Measuremnt Jacobian matrix H ***/
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i), ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    MatrixXd K(DIM_STATE, effct_feat_num_);
    // auto &&Hsub_T = Hsub.transpose();
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    // fout_dbg<<"HTz: "<<HTz<<endl;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;

    if (!degen_done_this_frame_ && config_setting_.Degeneracy_mitigation) {
        Degeneracy_detection(H_T_H.block<6, 6>(0, 0), last_lio_update_time, config_setting_.Degeneracy_Trans_thres, config_setting_.Degeneracy_Rot_thres, degen);
        degen_done_this_frame_ = true;
    }

    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    VD(DIM_STATE)
    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

    if (degen.any && config_setting_.Degeneracy_mitigation)
    {
        //硬抑制，效果不好？
        //suppressUpdateByDegeneracy(H_T_H.block<6, 6>(0, 0), degen, solution.head<6>(), last_lio_update_time);

        //Zhangji方案，将对应退化方向的特征向量置0
        suppressByEigMask(H_T_H.block<6, 6>(0, 0), degen, solution.head<6>(), last_lio_update_time);
    }

    int minRow, minCol;
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { flg_EKF_converged = true; }
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) { rematch_num++; }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      // _state.cov = (I_STATE - G) * _state.cov;
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      // total_distance += (_state.pos_end - position_last).norm();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

      // VD(DIM_STATE) K_sum  = K.rowwise().sum();
      // VD(DIM_STATE) P_diag = _state.cov.diagonal();
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) / scan_count + build_residual_time / scan_count;
  // ave_ekf_time = ave_ekf_time * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: " << build_residual_time << "s" << endl;
  // cout << "[ Mapping ] ave_ekf_time: " << ave_ekf_time << "s, ave_build_residual_time: " << ave_build_residual_time << "s" << endl;
}

void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void VoxelMapManager::BuildVoxelMap()
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  std::vector<pointWithVar> input_points;

  for (size_t i = 0; i < feats_down_world_->size(); i++)
  {
    pointWithVar pv;
    pv.point_w << feats_down_world_->points[i].x, feats_down_world_->points[i].y, feats_down_world_->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state_.cov.block<3, 3>(3, 3);
    pv.var = var;
    input_points.push_back(pv);
  }

  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
//      voxel_map_[position]->quater_length_ = voxel_size / 2;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->init_octo_tree();
  }
}

V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)
{
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}

//原始的更新体素地图方式，使用MATSLIDINGM模式和其匹配
//void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points)
//{
//  float voxel_size = config_setting_.max_voxel_size_;
//  float planer_threshold = config_setting_.planner_threshold_;
//  int max_layer = config_setting_.max_layer_;
//  int max_points_num = config_setting_.max_points_num_;
//  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
//  uint plsize = input_points.size();
//  for (uint i = 0; i < plsize; i++)
//  {
//    const pointWithVar p_v = input_points[i];
//    float loc_xyz[3];
//    for (int j = 0; j < 3; j++)
//    {
//      loc_xyz[j] = p_v.point_w[j] / voxel_size;
//      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
//    }
//    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
//    auto iter = voxel_map_.find(position);
//    if (iter != voxel_map_.end()) { voxel_map_[position]->UpdateOctoTree(p_v); }
//    else
//    {
//      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
//      voxel_map_[position] = octo_tree;
//      voxel_map_[position]->layer_init_num_ = layer_init_num;
//      voxel_map_[position]->quater_length_ = voxel_size / 4;
//      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
//      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
//      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
//      voxel_map_[position]->UpdateOctoTree(p_v);
//    }
//  }
//}

void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar>& input_points)
{
    const float voxel_size        = config_setting_.max_voxel_size_;
    const float planer_threshold  = config_setting_.planner_threshold_;
    const int   max_layer         = config_setting_.max_layer_;
    const int   max_points_num    = config_setting_.max_points_num_;
    const auto  layer_init_num    = config_setting_.layer_init_num_;

    for (const auto& p_v : input_points)
    {
        // 用 floor 处理负坐标更稳妥
        const int64_t vx = static_cast<int64_t>(std::floor(p_v.point_w[0] / voxel_size));
        const int64_t vy = static_cast<int64_t>(std::floor(p_v.point_w[1] / voxel_size));
        const int64_t vz = static_cast<int64_t>(std::floor(p_v.point_w[2] / voxel_size));
        const VOXEL_LOCATION pos{vx, vy, vz};

        auto it = lru_index_.find(pos);
        if (it != lru_index_.end())
        {
            // 命中：更新树 + 移动到 LRU 头
            CacheIter node = it->second;
            VoxelOctoTree* tree = node->second;     // list 里存的是 (pos, ptr)
            tree->UpdateOctoTree(p_v);
            voxel_map_cache_.splice(voxel_map_cache_.begin(), voxel_map_cache_, node);
        }
        else
        {
            // 未命中：创建体素树
            auto* tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0],
                                           max_points_num, planer_threshold);
            // 这里如果成员语义是“半边长”，应该设为 voxel_size/2，而不是 /4
            tree->quater_length_    = voxel_size / 4.0f;
            tree->voxel_center_[0] = (0.5f + static_cast<float>(pos.x)) * voxel_size;
            tree->voxel_center_[1] = (0.5f + static_cast<float>(pos.y)) * voxel_size;
            tree->voxel_center_[2] = (0.5f + static_cast<float>(pos.z)) * voxel_size;
            tree->temp_points_.push_back(p_v);
            tree->new_points_++;
            tree->layer_init_num_ = layer_init_num;

            // 插入 LRU 头 + 建立两张表
            voxel_map_cache_.emplace_front(pos, tree);
            lru_index_[pos] = voxel_map_cache_.begin();
            voxel_map_[pos]  = tree;   // 兼容旧接口：坐标 -> 指针
            ++stat_voxel_created_;
        }
    }

    auto world2voxel = [&](const Eigen::Vector3d& pw)->VOXEL_LOCATION{
        const double s = config_setting_.max_voxel_size_;
        return { (int64_t)std::floor(pw.x()/s),
                 (int64_t)std::floor(pw.y()/s),
                 (int64_t)std::floor(pw.z()/s) };
    };

    const double s = config_setting_.max_voxel_size_;
    double r_keep_m = 10.0;             // 建议做成ROS参数/配置，城市 8~15m
    double ahead_m  = 8.0;              // 10~15m
    double back_m   = 4.0;              // 4~8m

    const int r_keep = std::max(1, int(std::ceil(r_keep_m / s)));

    std::unordered_set<VOXEL_LOCATION, VoxelLocHash, VoxelLocEq> keep_set;
    VOXEL_LOCATION c = world2voxel(state_.pos_end);

// 球形邻域（高度很薄）
    for (int dx=-r_keep; dx<=r_keep; ++dx)
        for (int dy=-r_keep; dy<=r_keep; ++dy)
            for (int dz=-1; dz<=1; ++dz) {
                if (dx*dx + dy*dy + dz*dz > r_keep*r_keep) continue;
                keep_set.insert({c.x+dx, c.y+dy, c.z+dz});
            }

// 前/后向（水平面）
    Eigen::Vector3d v = state_.vel_end;
    if (v.norm() < 1e-3) v = Eigen::Vector3d::UnitX();
    v.normalize();
    Eigen::Vector3d right(-v.y(), v.x(), 0.0); right.normalize();
    int ax = int(std::ceil(ahead_m / s));
    int bx = int(std::ceil(back_m / s));
    int w  = r_keep;
    for (int i=-bx; i<=ax; ++i)
        for (int j=-w; j<=w; ++j) {
            Eigen::Vector3d off = i*v + j*right;
            keep_set.insert({ c.x + (int64_t)std::llround(off.x()),
                              c.y + (int64_t)std::llround(off.y()),
                              c.z });
        }

// 2) 淘汰（>= HIGH，先尽量避开 keep_set，必要时强制）
    const size_t HIGH = config_setting_.CAPACITY;
    const size_t LOW  = (HIGH>10) ? (HIGH*9/10) : 0;
    uint64_t evicted_this_call = 0;

    if (voxel_map_cache_.size() >= HIGH) {
        // 2.1 先删不在 keep_set 的
        while (voxel_map_cache_.size() > HIGH) {
            bool evicted = false;
            for (auto it = std::prev(voxel_map_cache_.end()); ; ) {
                const VOXEL_LOCATION key = it->first;
                if (keep_set.find(key) == keep_set.end()) {
                    VoxelOctoTree* ptr = it->second;
                    voxel_map_.erase(key);
                    lru_index_.erase(key);
                    voxel_map_cache_.erase(it);
                    delete ptr;
                    ++evicted_this_call;
                    ++stat_voxel_evicted_;
                    evicted = true;
                    break;
                }
                if (it == voxel_map_cache_.begin()) break;
                --it;
            }
            if (!evicted) break; // 全被keep挡住，跳出去做兜底
        }

        // 2.2 兜底：忽略keep_set，强制从尾删到 ≤ LOW
        if (voxel_map_cache_.size() > LOW) {
            while (voxel_map_cache_.size() > LOW) {
                auto it = std::prev(voxel_map_cache_.end());
                const VOXEL_LOCATION key = it->first;
                VoxelOctoTree* ptr = it->second;
                voxel_map_.erase(key);
                lru_index_.erase(key);
                voxel_map_cache_.erase(it);
                delete ptr;
                ++evicted_this_call;
                ++stat_voxel_evicted_;
            }
        }
    }

    last_evicted_ = evicted_this_call;
    ROS_INFO_STREAM_THROTTLE(1.0,
                             "[VoxelMap] cap=" << HIGH
                                               << " cache=" << voxel_map_cache_.size()
                                               << " keep="  << keep_set.size()
                                               << " evicted=" << evicted_this_call
                                               << " created_total=" << stat_voxel_created_.load()
                                               << " evicted_total=" << stat_voxel_evicted_.load());
}


void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
{
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_ptpl[i] = false;
  }
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < index.size(); i++)
  {
    pointWithVar &pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      VoxelOctoTree *current_octo = iter->second;
      PointToPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);
      if (!is_sucess)
      {
        VOXEL_LOCATION near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { near_position.x = near_position.x + 1; }
        else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { near_position.x = near_position.x - 1; }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { near_position.y = near_position.y + 1; }
        else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { near_position.y = near_position.y - 1; }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { near_position.z = near_position.z + 1; }
        else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { near_position.z = near_position.z - 1; }
        auto iter_near = voxel_map_.find(near_position);
        if (iter_near != voxel_map_.end()) { build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl); }
      }
      if (is_sucess)
      {
        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      }
      else
      {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++)
  {
    if (useful_ptpl[i]) { ptpl_list.push_back(all_ptpl_list[i]); }
  }
}

void VoxelMapManager::build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                                            double &prob, PointToPlane &single_ptpl)
{
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_w;
  if (current_octo->plane_ptr_->is_plane_)
  {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius_)
    {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
      sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;
      if (dis_to_plane < sigma_num * sqrt(sigma_l))
      {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob)
        {
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        }
        return;
      }
      else
      {
        // is_sucess = false;
        return;
      }
    }
    else
    {
      // is_sucess = false;
      return;
    }
  }
  else
  {
    if (current_layer < max_layer)
    {
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      {
        if (current_octo->leaves_[leafnum] != nullptr)
        {

          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
        }
      }
      return;
    }
    else { return; }
  }
}

void VoxelMapManager::pubVoxelMap()
{
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) { trace = max_trace; }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  voxel_map_pub_.publish(voxel_plane);
  loop.sleep();
}

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)
{
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::Quaternion &q)
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void VoxelMapManager::mapSliding()
{
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  //get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//discrete global
  clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size, (int64_t)loc_xyz[0] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[1] + config_setting_.half_map_size, (int64_t)loc_xyz[1] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[2] + config_setting_.half_map_size, (int64_t)loc_xyz[2] - config_setting_.half_map_size);
  //clearAll();
  double t_sliding_end = omp_get_wtime();
  std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}

void VoxelMapManager::clearAll()
{
    // delete each tree exactly once by walking the list
    for (auto& kv : voxel_map_cache_) {
        delete kv.second;
    }
    voxel_map_cache_.clear();
    voxel_map_.clear();
    lru_index_.clear();

    // force capacity to release
    CacheList().swap(voxel_map_cache_);
    decltype(voxel_map_)().swap(voxel_map_);
    decltype(lru_index_)().swap(lru_index_);
}

void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
    int delete_voxel_count = 0;
    for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
    {
        const VOXEL_LOCATION& loc = it->first;
        bool should_remove = (loc.x > x_max || loc.x < x_min ||
                              loc.y > y_max || loc.y < y_min ||
                              loc.z > z_max || loc.z < z_min);
        if (should_remove)
        {
            // remove from list+lru_index+map in this order
            auto it_lru = lru_index_.find(loc);
            if (it_lru != lru_index_.end()) {
                CacheIter node = it_lru->second;
                VoxelOctoTree* ptr = node->second;
                voxel_map_cache_.erase(node);
                lru_index_.erase(it_lru);
                delete ptr;
            } else {
                // fall back to map pointer (shouldn't happen if indices are consistent)
                delete it->second;
            }
            it = voxel_map_.erase(it);
            delete_voxel_count++;
        }
        else
        {
            ++it;
        }
    }
    std::cout << RED << "[DEBUG]: Delete " << delete_voxel_count << " root voxels" << RESET << "\\n";
}