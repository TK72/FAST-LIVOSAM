/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

VIOManager::VIOManager()
{
    // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager() {
    // 先清掉回收桶，避免与下面的遍历重复 delete
    drainFeatGarbage();

    delete visual_submap;

    for (auto& kv : warp_map) delete kv.second;
    warp_map.clear();

    // 剩余仍留在 feat_map 里的体素统一删除（不在回收桶内的那些）
    for (auto& kv : feat_map) {
        if (feat_garbage_set_.count(kv.second) == 0) {
            delete kv.second;
        }
    }
    feat_map.clear();

    feat_lru_index_.clear();
    feat_lru_.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;

  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

//void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
//{
//  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
//  double voxel_size = 0.5;
//  float loc_xyz[3];
//  for (int j = 0; j < 3; j++)
//  {
//    loc_xyz[j] = pt_w[j] / voxel_size;
//    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
//  }
//  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
//  auto iter = feat_map.find(position);
//  if (iter != feat_map.end())
//  {
//    iter->second->voxel_points.push_back(pt_new);
//    iter->second->count++;
//  }
//  else
//  {
//    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
//    ot->voxel_points.push_back(pt_new);
//    feat_map[position] = ot;
//  }
//}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
    // 把点放入对应视觉体素
    const V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    const double voxel_size = FEAT_VOXEL_SIZE_; // 1.0m
    int64_t ix = static_cast<int64_t>(std::floor(pt_w[0] / voxel_size));
    int64_t iy = static_cast<int64_t>(std::floor(pt_w[1] / voxel_size));
    int64_t iz = static_cast<int64_t>(std::floor(pt_w[2] / voxel_size));
    const VOXEL_LOCATION position(ix, iy, iz);

    auto it = feat_map.find(position);
    if (it != feat_map.end()) {
        // 命中：直接追加
        it->second->voxel_points.push_back(pt_new);
        it->second->count++;
        // LRU：移到头
        touchFeatLRU_(position, it->second);
    } else {
        // 未命中：新建体素
        VOXEL_POINTS *ot = new VOXEL_POINTS(0);
        ot->voxel_points.push_back(pt_new);
        ot->count = 1;
        feat_map[position] = ot;
        // LRU：插入头
        touchFeatLRU_(position, ot);
    }
//    cout<<FEAT_CAP_HIGH_<<endl;
    evictFeatLRU_Oldest_(FEAT_CAP_HIGH_, /*low_cap*/ 0);
    // 容量控制（延迟回收 + 去重）
//    if (feat_lru_.size() > FEAT_CAPACITY) {
////        cout<<"feat_map run"<<endl;
//        const std::size_t LOW = (FEAT_CAPACITY > 10) ? (FEAT_CAPACITY * 9 / 10) : 0;
//
//        while (feat_lru_.size() > LOW) {
//            if (feat_lru_.empty()) break;
//
//            bool evicted = false;
//            // 从尾部开始找可淘汰的节点
//            for (auto itl = std::prev(feat_lru_.end()); ; ) {
//                const VOXEL_LOCATION key = itl->first;
//                VOXEL_POINTS* vp         = itl->second;
//
//                // 如果你维护了保留集 keep_set（FOV/在用体素），这里跳过它：
//                // if (keep_set.count(key)) {
//                //   if (itl == feat_lru_.begin()) break; else { --itl; continue; }
//                // }
//
//                // 1) 先从映射结构中摘掉（只 erase，不 delete）
//                feat_lru_index_.erase(key);
//                feat_map.erase(key);
//
//                // 2) 延迟回收：去重后再入桶（关键修复：避免 double free）
//                if (feat_garbage_set_.insert(vp).second) {
//                    feat_garbage_.push_back(vp);
//                }
//
//                // 3) 删链表节点
//                itl = feat_lru_.erase(itl);
//
//                evicted = true;
//                break; // 本轮淘汰一个就够
//
//                // 若需要继续向前：
//                // if (itl == feat_lru_.begin()) break; else --itl;
//            }
//
//            if (!evicted) break; // 全部都是保留集，退出避免死循环
//        }
//    }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = FEAT_VOXEL_SIZE_;

  if (!normal_en) warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map)
  {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      touchFeatLRU_(position, corre_voxel->second);
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i])
      {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          touchFeatLRU_(sample_pos, corre_feat_map->second);
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8);
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_)
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (normal_en && !pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
      else
      {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();

        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();

        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      // 仅在开启半稠密/鲁棒时启用限深+权重
      float w_depth = 1.0f;
      if (semidense_en) {
            V3D pt_cam(new_frame_->w2f(pt->pos_));
            float z_now = pt_cam[2];
            if (z_now > depth_max_m_) continue; // 限深：超远点丢弃
            w_depth = std::min(1.0f, (float)(depth_ref_m_ / std::max(1e-3f, z_now)));
      }

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);

      if (semidense_en) visual_submap->weights.push_back(w_depth);
    }
  }
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);

    if (semidense_en) {
        // 已有：cv::Mat depth_img (float), img(灰度), width/height, border, grid_n_width/height, grid_size
        cv::Mat grad;
        cv::Mat gx, gy;
        cv::Sobel(img, gx, CV_32F, 1, 0, 3);
        cv::Sobel(img, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, grad);

        // 半稠密候选
        const int K_PER_TILE = 80;             // 每 tile 上限，按实时性调
        std::vector <cv::Point> candidates;
        candidates.reserve(5000);

        for (int gr = 0; gr < grid_n_height; ++gr) {
            for (int gc = 0; gc < grid_n_width; ++gc) {
                int u0 = gc * grid_size, v0 = gr * grid_size;
                int u1 = std::min(u0 + grid_size, width - 1);
                int v1 = std::min(v0 + grid_size, height - 1);

                // 计算这个 tile 的分位阈值（80~90分位）
                std::vector<float> buf;
                buf.reserve((u1 - u0) * (v1 - v0));
                for (int v = v0; v < v1; ++v)
                    for (int u = u0; u < u1; ++u)
                        if (depth_img.at<float>(v, u) > 0) buf.push_back(grad.at<float>(v, u));
                //这是一个很简单的判断，如果像素的数量小于50 就不会进行下面的半稠密特征点提取
                //            if (buf.size() < 50) continue;
//            std::nth_element(buf.begin(), buf.begin()+buf.size()*8/10, buf.end());
//            float thr = buf[buf.size()*8/10];

                int valid = (int) buf.size();
                if (valid == 0) continue;

                // 动态分位：有效数少时，降低分位；再加一个绝对下限门槛
                int pct = 70;
                if (valid < 200) pct = 60;
                if (valid < 80)  pct = 50;
                if (valid < 30)  pct = 30;

                int kth = std::max(0, std::min(valid - 1, valid * pct / 100));
                std::nth_element(buf.begin(), buf.begin() + kth, buf.end());
                float thr = buf[kth];

                // 绝对阈值下限（远景梯度普遍小，避免阈值过高）
                thr = std::max(thr, 10.0f);

                // NMS + 选 Top-K
                std::vector <std::pair<float, cv::Point>> topk;
                topk.reserve(K_PER_TILE * 2);
                for (int v = v0 + 1; v < v1 - 1; ++v)
                    for (int u = u0 + 1; u < u1 - 1; ++u) {
                        float g = grad.at<float>(v, u);
                        if (g < thr || depth_img.at<float>(v, u) <= 0) continue;
                        // 简单3x3极大值
                        bool is_max = true;
                        for (int dv = -1; dv <= 1 && is_max; ++dv)
                            for (int du = -1; du <= 1; ++du)
                                if (grad.at<float>(v + dv, u + du) > g) {
                                    is_max = false;
                                    break;
                                }
                        if (!is_max) continue;
                        topk.emplace_back(g, cv::Point(u, v));
                    }
                std::partial_sort(topk.begin(), topk.begin() + std::min((int) topk.size(), K_PER_TILE), topk.end(),
                                  [](auto &a, auto &b) { return a.first > b.first; });
                int keep = std::min((int) topk.size(), K_PER_TILE);
                for (int i = 0; i < keep; i++) candidates.push_back(topk[i].second);
            }
        }

        // 把本帧的候选像素存下来，后面在 processFrame 里统一画
        semidense_candidates_.clear();
        semidense_candidates_.insert(semidense_candidates_.end(),
                                     candidates.begin(), candidates.end());

        // === 将 candidates 变成“临时 VisualPoint”，并把所有并行数组一并补齐 ===
        const Matrix2d A_I = Matrix2d::Identity();   // 参考-当前同一帧，用单位仿射即可
        for (auto &ptpx: candidates) {
            const int u = ptpx.x, v = ptpx.y;

            // 1) 边界/深度检查
            if (u < border || v < border || u >= width - border || v >= height - border) continue;
            float d = depth_img.at<float>(v, u);
            if (d <= 0) continue;
            if (d > depth_max_m_) continue; // NEW: 半稠密限深，不入图

            // 2) 反投影到世界坐标
            V3D xyz = cam->cam2world(V2D(u, v));
            xyz *= d / xyz[2];
            V3D p_w = new_frame_->f2w(xyz);

            // 3) 构造参考补丁（参考帧=当前帧）
            float *patch = new float[patch_size_total];
            getImagePatch(img, V2D(u, v), patch, 0);

            // 4) 构造 VisualPoint + Feature（参考帧设为当前帧）
            semidense_pool_.emplace_back(new VisualPoint(p_w));
            VisualPoint *vp = semidense_pool_.back().get();
            Feature *ftr = new Feature(vp, patch, V2D(u, v), cam->cam2world(V2D(u, v)),
                                       new_frame_->T_f_w_, /*level_ref*/0);
            ftr->img_ = img;                    // 参考图像
            ftr->id_ = new_frame_->id_;        // 用当前帧ID标识
            ftr->inv_expo_time_ = state->inv_expo_time;   // 参考曝光（当前帧）

            vp->addFrameRef(ftr);
            vp->is_normal_initialized_ = false;            // 让后续走 normal_en=false 的分支
            vp->ref_patch = ftr;
            vp->has_ref_patch_ = true;

            // 5) 预先生成各金字塔层的 warp_patch（参考->当前，同一帧，相当于单位变换）
            std::vector<float> patch_wrap(warp_len, 0.f);
            int search_level = getBestSearchLevel(A_I, /*max*/2); // 通常=0
            for (int pyr = 0; pyr <= patch_pyrimid_level - 1; ++pyr) {
                warpAffine(A_I, ftr->img_, ftr->px_, /*level_ref*/0,
                           search_level, pyr, patch_size_half, patch_wrap.data());
            }

            // 6) 计算一个粗略光度误差（可选）
            getImagePatch(img, V2D(u, v), patch_buffer.data(), 0);
            float err = 0.f;
            for (int k = 0; k < patch_size_total; ++k) {
                float refv = ftr->inv_expo_time_ * patch_wrap[k];
                float curv = state->inv_expo_time * patch_buffer[k];
                float r = refv - curv;
                err += r * r;
            }

            // 7) 不再 push 到 visual_submap，本帧不参与优化；改为挂起，优化后入图
            semidense_pending_.push_back(vp);
        }

        //border检查，防止维度崩溃！
        {
            auto &S = *visual_submap;
            size_t n = S.voxel_points.size();
            size_t m = n;
            m = std::min(m, S.search_levels.size());
            m = std::min(m, S.warp_patch.size());
            m = std::min(m, S.inv_expo_list.size());
            m = std::min(m, S.errors.size());
            m = std::min(m, S.propa_errors.size());
            if (semidense_en) m = std::min(m, S.weights.size()); // 只有开启时才对齐权重

            if (m < n) {
                S.voxel_points.resize(m);
                S.search_levels.resize(m);
                S.warp_patch.resize(m);
                S.inv_expo_list.resize(m);
                S.errors.resize(m);
                S.propa_errors.resize(m);
                if (semidense_en) S.weights.resize(m);
            }
            total_points = (int) m;
        }

        if ((int) visual_submap->voxel_points.size() > max_points_cap_) {
            struct Item {
                float z;
                int i;
            };
            std::vector <Item> order;
            order.reserve(visual_submap->voxel_points.size());

            for (int i = 0; i < (int) visual_submap->voxel_points.size(); ++i) {
                VisualPoint *pt = visual_submap->voxel_points[i];
                float z = (new_frame_->w2f(pt->pos_))[2];
                order.push_back({z, i});
            }
            std::stable_sort(order.begin(), order.end(),
                             [](const Item &a, const Item &b) { return a.z < b.z; });

            // 选择前 K
            int K = max_points_cap_;
            std::vector<int> keep;
            keep.reserve(K);
            for (int k = 0; k < K; ++k) keep.push_back(order[k].i);
            std::sort(keep.begin(), keep.end()); // 便于按序拣取

            auto pick = [&](auto &vec) {
                using T = typename std::decay<decltype(vec)>::type::value_type;
                std::vector <T> out;
                out.reserve(keep.size());
                for (int idx: keep) out.push_back(vec[idx]);
                vec.swap(out);
            };

            auto &S = *visual_submap;
            pick(S.voxel_points);
            pick(S.search_levels);
            pick(S.warp_patch);
            pick(S.inv_expo_list);
            pick(S.errors);
            pick(S.propa_errors);
            pick(S.weights); // NEW

            total_points = (int) S.voxel_points.size();
        }

        total_points = visual_submap->voxel_points.size();
        printf("[ VIO ] Semi-dense add: %zu, total points: %d\n",
               candidates.size(), total_points);
    }
}

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;

  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pg[i].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }

      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);

    // ---- 触发一次 LRU 淘汰，保证视觉体素不超上限 ----
    {
// 与 updateState 中一致的位姿计算
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        M3D Rcw = Rci * Rwi.transpose();
        V3D cam_pos_W = Pwi; // 相机中心（世界）
        V3D forward_cam(0,0,1); // 相机坐标系前向 z
        V3D forward_W = Rcw.transpose() * forward_cam; // 转到世界系


        evictFeatLRU_(cam_pos_W, forward_W);
    }
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;

    float *patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / FEAT_VOXEL_SIZE_;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);

            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr) continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float w_point = 1.0f;
      if (semidense_en && i < (int)visual_submap->weights.size())
         w_point = std::max(1e-6f, visual_submap->weights[i]);

      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];

          //进行huber核加权，关闭时不加权
          double a = std::fabs(res);
          double w_huber = (!semidense_en) ? 1.0 : (a <= huber_delta_ ? 1.0 : (huber_delta_ / a));
          double w = semidense_en ? (w_point * w_huber) : 1.0;  // 关时完全不加权

          z(i * patch_size_total + x * patch_size + y) = w * res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) *= w; // NEW
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();

    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);

    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
        float w_point = 1.0f;
        if (semidense_en && i < (int)visual_submap->weights.size())
            w_point = std::max(1e-6f, visual_submap->weights[i]);
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          double a = std::fabs(res);
          double w_huber = (!semidense_en) ? 1.0 : (a <= huber_delta_ ? 1.0 : (huber_delta_ / a));
          double w = semidense_en ? (w_point * w_huber) : 1.0;  // 关时完全不加权

          // “加权等价观测”：z、H 同时乘以同一个权重
          z(i * patch_size_total + x * patch_size + y) = w * res;

          patch_error += res * res;
          n_meas += 1;

            if (exposure_estimate_en) {
                H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value;
                H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) *= w; // NEW
            } else {
                H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
                H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) *= w; // NEW
            }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();
  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);

  resetGrid();

  double t1 = omp_get_wtime();

  retrieveFromVisualSparseMap(img, pg, feat_map);

  //挑选高价值点
  //pruneVisualSubmap_(img);

  double t2 = omp_get_wtime();

  computeJacobianAndUpdateEKF(img);

    // === 把本帧创建的半稠密点插入到地图体素（供下一帧使用）当打开semidense_en时才会实现 ===
    if (semidense_en && !semidense_pending_.empty()) {
        for (auto* vp : semidense_pending_) {
            insertPointIntoVoxelMap(vp);  // 地图开始接管这块内存
        }
        // 释放 unique_ptr 所有权并清空
        for (auto &up : semidense_pool_) up.release();
        semidense_pool_.clear();
        semidense_pending_.clear();
    }

  double t3 = omp_get_wtime();

  generateVisualMapPoints(img, pg);

  double t4 = omp_get_wtime();

  plotTrackedPoints();

    // 把半稠密黄点画到 img_cp 上（显示用 + 第二张保存用）
    if (semidense_en && !semidense_candidates_.empty())
    {
        for (const auto& p : semidense_candidates_) {
            cv::circle(img_cp, p, 3, cv::Scalar(0, 255, 255), -1, 2); // 黄点
        }
    }

  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;

  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;

  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  drainFeatGarbage();
  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
  // cv::imwrite("/home/chunran/Desktop/raycasting/" + std::to_string(new_frame_->id_) + ".png", img_cp);
}

void VIOManager::touchFeatLRU_(const VOXEL_LOCATION& key, VOXEL_POINTS* vp) {
    auto it = feat_lru_index_.find(key);
    if (it != feat_lru_index_.end()) {
        // 命中：移到表头
        feat_lru_.splice(feat_lru_.begin(), feat_lru_, it->second);
        it->second = feat_lru_.begin();
    } else {
        // 新插入：放表头并登记迭代器
        feat_lru_.emplace_front(key, vp);
        feat_lru_index_[key] = feat_lru_.begin();
    }
}


// 释放一个体素内的所有 VisualPoint，并返回释放点的数量
static size_t FreeVoxel_AllPoints_(VOXEL_POINTS* v)
{
    if (!v) return 0;
    size_t n = 0;
    for (auto* p : v->voxel_points) {
        if (!p) continue;
        delete p; // 交由 VisualPoint 析构释放其内部 Feature
        ++n;
    }
    delete v;
    return n;
}


void VIOManager::evictFeatLRU_(const Eigen::Vector3d& cam_pos_W,
                               const Eigen::Vector3d& forward_W)
{
    last_evict_voxels_ = last_evict_points_ = 0;

    size_t high = (FEAT_CAP_HIGH_ > 0) ? FEAT_CAP_HIGH_ : (FEAT_CAPACITY > 0 ? FEAT_CAPACITY : 5000);
    size_t low  = (FEAT_CAP_LOW_  > 0) ? FEAT_CAP_LOW_  : (high > 10 ? high * 9 / 10 : 0);
    if (feat_lru_.size() <= high) return;

    auto world2voxel = [&](const Eigen::Vector3d& pw)->VOXEL_LOCATION{
        const double s = FEAT_VOXEL_SIZE_;
        return { (int64_t)std::floor(pw.x()/s),
                 (int64_t)std::floor(pw.y()/s),
                 (int64_t)std::floor(pw.z()/s) };
    };

    // ---------- 构造保留集 ----------
    double keep_r   = (FEAT_KEEP_RADIUS_M_ > 0) ? FEAT_KEEP_RADIUS_M_ : 6.0;
    int    keep_z   = (FEAT_KEEP_Z_        > 0) ? FEAT_KEEP_Z_        : 1;
    double keep_a   = (FEAT_KEEP_AHEAD_M_  > 0) ? FEAT_KEEP_AHEAD_M_  : 9.0;
    double keep_b   = (FEAT_KEEP_BACK_M_   > 0) ? FEAT_KEEP_BACK_M_   : 3.0;

//    cout<< FEAT_CAP_HIGH_ <<" "<<FEAT_CAP_LOW_<<endl;

    const double s = FEAT_VOXEL_SIZE_;
    const int r_keep = std::max(1, (int)std::ceil(FEAT_KEEP_RADIUS_M_ / s));
    VOXEL_LOCATION c = world2voxel(cam_pos_W);

    std::unordered_set<VOXEL_LOCATION> keep_set;
    keep_set.reserve((size_t)(2*r_keep+1)*(2*r_keep+1)*(2*FEAT_KEEP_Z_+1));

    // 立方体邻域
    for (int dx=-r_keep; dx<=r_keep; ++dx)
        for (int dy=-r_keep; dy<=r_keep; ++dy)
            for (int dz=-FEAT_KEEP_Z_; dz<=FEAT_KEEP_Z_; ++dz)
                keep_set.insert({c.x+dx, c.y+dy, c.z+dz});

    // 前/后向走廊（水平面）
    Eigen::Vector3d v = forward_W;
    v.z() = 0;
    if (v.norm() < 1e-3) v = Eigen::Vector3d::UnitX();
    v.normalize();
    Eigen::Vector3d right(-v.y(), v.x(), 0.0);
    const int ax = (int)std::ceil(FEAT_KEEP_AHEAD_M_ / s);
    const int bx = (int)std::ceil(FEAT_KEEP_BACK_M_ / s);
    const int w  = r_keep;
    for (int i=-bx; i<=ax; ++i) {
        for (int j=-w; j<=w; ++j) {
            Eigen::Vector3d off = i*v + j*right;
            keep_set.insert({ c.x+(int64_t)std::llround(off.x()),
                              c.y+(int64_t)std::llround(off.y()),
                              c.z });
        }
    }

    // ---------- 淘汰：从 LRU 尾部开始，优先删不在保留集里的 ----------
    const size_t LOW = FEAT_CAP_LOW_;
    while (feat_lru_.size() > LOW) {
        bool evicted = false;

        for (auto it = std::prev(feat_lru_.end()); ; ) {
            const VOXEL_LOCATION key = it->first;
            if (keep_set.find(key) == keep_set.end()) {
                VOXEL_POINTS* vp = it->second;

                feat_lru_index_.erase(key);
                feat_map.erase(key);
                it = feat_lru_.erase(it);

                if (feat_garbage_set_.insert(vp).second) {
                    feat_garbage_.push_back(vp);
                }
                ++last_evict_voxels_;
                last_evict_points_ += vp ? vp->voxel_points.size() : 0;
                evicted = true;
                break;
            }
            if (it == feat_lru_.begin()) break;
            --it;
        }

        if (!evicted) {
            // 兜底：全部都在保留集里，还是要降到 LOW -> 强制删一个最旧的
            auto it = std::prev(feat_lru_.end());
            const VOXEL_LOCATION key = it->first;
            VOXEL_POINTS* vp         = it->second;

            feat_lru_index_.erase(key);
            feat_map.erase(key);
            feat_lru_.erase(it);

            if (feat_garbage_set_.insert(vp).second) {
                feat_garbage_.push_back(vp);
            }
            ++last_evict_voxels_;
            last_evict_points_ += vp ? vp->voxel_points.size() : 0;
        }
    }

    if (last_evict_voxels_ > 0) {
        printf("\033[1;33m[ VIO ][LRU] Evicted %zu voxels (~%zu points). LRU=%zu\033[0m\n",
               last_evict_voxels_, last_evict_points_, feat_lru_.size());
    }
}

void VIOManager::drainFeatGarbage() {
    for (auto* vp : feat_garbage_) {
        delete vp;                   // 释放体素；其中的 VisualPoint 析构会释放 Feature
    }
    feat_garbage_.clear();
    feat_garbage_set_.clear();
}


void VIOManager::evictFeatLRU_Oldest_(size_t high_cap, size_t low_cap)
{
    if (feat_lru_.size() <= high_cap) return;
    const size_t LOW = (low_cap > 0 ? low_cap :
                        (high_cap > 10 ? (high_cap * 9 / 10) : 0));

    while (feat_lru_.size() > LOW) {
        if (feat_lru_.empty()) break;
        auto it = std::prev(feat_lru_.end());                 // 从尾部挑最久未用
        const VOXEL_LOCATION key = it->first;
        VOXEL_POINTS* vp         = it->second;

        // 1) 从“主表 & 索引”摘掉
        feat_lru_index_.erase(key);
        feat_map.erase(key);
        feat_lru_.erase(it);

        // 2) 延迟回收（去重防重删）
        if (feat_garbage_set_.insert(vp).second) {
            feat_garbage_.push_back(vp);
        }
        ++last_evict_voxels_;
        last_evict_points_ += vp ? vp->voxel_points.size() : 0;
    }

    if (last_evict_voxels_ > 0) {
        printf("\033[1;33m[ VIO ][LRU] Evicted %zu voxels (~%zu points). LRU=%zu\033[0m\n",
               last_evict_voxels_, last_evict_points_, feat_lru_.size());
    }
}

void VIOManager::pruneVisualSubmap_(const cv::Mat& img)
{
    auto& S = *visual_submap;
    const int N = (int)S.voxel_points.size();
    if (N == 0) return;

    // 预计算梯度图（一次）
    cv::Mat gx, gy, grad;
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, grad);

    struct Cand { float score; int idx; int tile; };
    std::vector<Cand> cands; cands.reserve(N);

    // 用于 tile 分桶
    std::vector<std::vector<Cand>> buckets(length);

    for (int i = 0; i < N; ++i) {
        VisualPoint* pt = S.voxel_points[i];
        if (!pt) continue;

        // 像素位置 & tile
        V2D pc(new_frame_->w2c(pt->pos_));
        if (!cam->isInFrame(pc.cast<int>(), border)) continue;
        const int tile = (int(pc[1]) / grid_size) * grid_n_width + (int(pc[0]) / grid_size);

        // 深度与限深
        const float z = (float)(new_frame_->w2f(pt->pos_)[2]);
        if (z <= min_depth_m_ || z > depth_max_m_) continue;

        // 计算 patch 上的梯度能量（简单均值/最大值均可，这里取均值）
        const int u0 = (int)pc[0] - patch_size_half;
        const int v0 = (int)pc[1] - patch_size_half;
        if (u0 < 1 || v0 < 1 || u0 + patch_size >= width-1 || v0 + patch_size >= height-1) continue;

        float g_sum = 0.f, g_max = 0.f;
        for (int dv = 0; dv < patch_size; ++dv) {
            const float* gp = grad.ptr<float>(v0 + dv) + u0;
            for (int du = 0; du < patch_size; ++du) {
                float g = gp[du];
                g_sum += g;
                if (g > g_max) g_max = g;
            }
        }
        const float grad_energy = g_sum / (float)patch_size_total;
        if (grad_energy < min_grad_) continue;

        // 轨迹寿命（越久越稳）
        const float w_track = std::min(1.f, (float)pt->obs_.size() / 8.f);

        // 视角权重（法线可用时避免掠射角）
        float w_view = 1.0f;
        if (pt->is_normal_initialized_) {
            V3D n_cam = new_frame_->T_f_w_.rotation_matrix() * pt->normal_;
            V3D dir   = (new_frame_->w2f(pt->pos_)).normalized();
            float c   = (float)std::abs(n_cam.dot(dir));            // [0,1]
            // 允许一定掠射，但对过小 cos 做抑制（0.2~1 映射到 0~1）
            w_view = std::max(0.f, std::min(1.f, (c - 0.2f) / 0.8f));
        }

        // 深度权重（偏好中近景）
        const float w_depth = std::min(1.f, (float)(depth_ref_m_ / std::max(1e-3f, z)));

        // 用传播误差做一个轻微惩罚（更喜欢误差小的）
        const float err = (i < (int)S.propa_errors.size()) ? (float)S.propa_errors[i] : 0.f;
        const float err_norm = err / (patch_size_total * 50.f);  // 尺度化，经验数
        const float pen      = 1.0f / (1.0f + err_norm);

        // 最终评分（可按需调权）
        const float score = grad_energy * w_depth * (0.2f + 0.8f * w_track) * w_view * pen;

        buckets[tile].push_back({score, i, tile});
    }

    // 每 tile 先取前 K
    std::vector<int> keep_idx;
    keep_idx.reserve(std::min(topk_total_, (int)S.voxel_points.size()));

    for (auto& B : buckets) {
        if (B.empty()) continue;
        const int k = std::min((int)B.size(), topk_per_tile_);
        std::partial_sort(B.begin(), B.begin() + k, B.end(),
                          [](const Cand& a, const Cand& b){ return a.score > b.score; });
        for (int i = 0; i < k; ++i) keep_idx.push_back(B[i].idx);
    }

    // 全局补齐到 TopK_total
    if ((int)keep_idx.size() < topk_total_) {
        std::vector<Cand> rest;
        rest.reserve(N);
        for (auto& B : buckets) {
            if ((int)B.size() > topk_per_tile_) {
                rest.insert(rest.end(), B.begin() + topk_per_tile_, B.end());
            }
        }
        std::sort(rest.begin(), rest.end(),
                  [](const Cand& a, const Cand& b){ return a.score > b.score; });

        // 去重填充
        std::vector<char> mark(N, 0);
        for (int id : keep_idx) mark[id] = 1;
        for (auto& c : rest) {
            if ((int)keep_idx.size() >= topk_total_) break;
            if (!mark[c.idx]) {
                keep_idx.push_back(c.idx);
                mark[c.idx] = 1;
            }
        }
    }

    // —— 收缩所有并行数组（保持索引一致）——
    auto pick_by = [&](auto& vec) {
        using T = typename std::decay<decltype(vec)>::type::value_type;
        if ((int)vec.size() == 0) return;
        std::vector<T> out; out.reserve(keep_idx.size());
        for (int id : keep_idx) out.push_back(vec[id]);
        vec.swap(out);
    };

    pick_by(S.voxel_points);
    if (!S.search_levels.empty()) pick_by(S.search_levels);
    if (!S.warp_patch.empty())    pick_by(S.warp_patch);
    if (!S.inv_expo_list.empty()) pick_by(S.inv_expo_list);
    if (!S.errors.empty())        pick_by(S.errors);
    if (!S.propa_errors.empty())  pick_by(S.propa_errors);
    if (semidense_en && !S.weights.empty()) pick_by(S.weights);

    total_points = (int)S.voxel_points.size();
    printf("[ VIO ] PruneByScore: keep %d / %d (Top=%d, per-tile=%d)\n",
           total_points, N, topk_total_, topk_per_tile_);
}
