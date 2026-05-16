#!/bin/bash
#
# run_rviz_minibus.sh
# 宿主机起 RViz，连到容器里的 roscore，并自动加载固定的 RViz 配置

#### 1. RViz 配置文件路径 ####
RVIZ_CONFIG="/home/liuminzhe/catkin_fastlivosam_dynamic_remove/fast_livo2.rviz"

#### 2. ROS 环境初始化 ####
# 确保宿主机上已经安装了 ROS Noetic
source /opt/ros/noetic/setup.bash

#### 3. 指向容器里的 roscore ####
# 你用的是 --network host 跑容器，所以容器里的 roscore 也是监听宿主机的 11311
export ROS_MASTER_URI="http://127.0.0.1:11311"

# 让 ROS 认为“我是谁”。宿主机上跑 rviz，用本机 IP 就行
export ROS_IP="127.0.0.1"

echo "[INFO] ROS_MASTER_URI = $ROS_MASTER_URI"
echo "[INFO] ROS_IP         = $ROS_IP"
echo "[INFO] RViz config    = $RVIZ_CONFIG"
echo "[INFO] Starting RViz ..."

#### 4. 启动 RViz 并加载配置 ####
rviz -d "$RVIZ_CONFIG"
