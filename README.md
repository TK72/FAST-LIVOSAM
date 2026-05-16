# FAST-LIVOSAM

<div align="center">

### A Robust Global LiDAR-Visual-Inertial SLAM Framework with CenterPoint-based 3D Object Detection and Dynamic Points Removal

</div>

---

# Overview

FAST-LIVOSAM is a robust global LiDAR-Visual-Inertial SLAM framework built upon FAST-LIVO2 with:

* Global pose graph optimization
* Dynamic object removal
* TensorRT-accelerated CenterPoint 3D object detection
* Multi-sensor fusion
* High-performance real-time mapping

The framework supports:

* LiDAR-Inertial-Visual SLAM
* Dynamic scene understanding
* Global optimization with GTSAM
* TensorRT deployment
* GPU acceleration

---

# Features

* FAST-LIVO based tightly-coupled LIO/VIO
* Global pose graph optimization
* Dynamic object removal
* TensorRT CenterPoint inference
* CUDA accelerated point cloud processing
* Multi-threaded optimization
* ROS-based deployment
* Docker support

---

# System Architecture

<p align="center">
  <img src="docs/framework.png" width="90%">
</p>

---

# Demo

Demo videos can be found in the `assets/` directory of this repository, including:

* CenterPoint 3D detection and Dynamic object removal
* Global mapping

---

# Dependencies

## Basic Environment

* Ubuntu 20.04
* ROS Noetic
* CUDA 11.x
* TensorRT 8.x
* C++17

---

## Required ROS Packages

```bash
sudo apt install ros-noetic-pcl-ros
sudo apt install ros-noetic-tf
sudo apt install ros-noetic-cv-bridge
sudo apt install ros-noetic-image-transport
sudo apt install ros-noetic-eigen-conversions
```

---

## Third-party Libraries

* Eigen3
* PCL
* OpenCV
* Boost
* Sophus (legacy version)
* GeographicLib
* GTSAM
* OpenMP
* CUDA
* TensorRT
* spconv

---

# Build

## Clone Repository

```bash
git clone https://github.com/TK72/FAST-LIVOSAM.git
cd FAST-LIVOSAM
```

---

## Build SLAM Only

```bash
catkin_make
```

---

## Build with CenterPoint

```bash
catkin_make --cmake-args -DBUILD_CENTERPOINT=ON
```

---

# Run

## Source Workspace

```bash
source devel/setup.bash
```

---

## Launch

```bash
roslaunch fast_livo mapping_GREAT_WHU_centerpoint.launch
```

---

# Dataset

Current experiments are mainly conducted on:

* GREAT-WHU
* KITTI

---

# TensorRT Notes

Before enabling CenterPoint:

* TensorRT must be installed
* CUDA must be properly configured
* spconv shared library must exist

Example:

```bash
export SPCONV_ROOT=/path/to/libspconv
export LD_LIBRARY_PATH=$SPCONV_ROOT/lib/x86_64:$LD_LIBRARY_PATH
```

---

# Citation

If you find this project useful, please consider citing our paper:

```bibtex
@article{fastlivosam,
  title={Robust Urban SLAM Via Resilient GNSS-IMU-LiDAR-Camera-Loop Fusion},
  author={Minzhe Liu and Hongjuan Zhang and Zhibo Zhao and Chengzhi Hong and Haoyu Wang and Zilong Xiao and Bijun Li},
  journal={Under review},
  year={2026}
}
```

---

# Acknowledgements

This project is built upon:

* FAST-LIVO2
* FAST-LIO2
* FAST-LIOSAM
* GTSAM
* Sophus
* CenterPoint
* TensorRT
* ROS

---

We would like to express our sincere gratitude to the authors for their valuable open-source contributions. Their excellent work and open-source spirit have provided important inspiration and support for the development of this project.
---

# License

MIT License
