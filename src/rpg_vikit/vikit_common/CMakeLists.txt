cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME vikit_common)
project(${PROJECT_NAME} CXX)
set(CMAKE_BUILD_TYPE Release)

# 编译选项
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CPU arch: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  add_compile_options(-mmmx -msse -msse2 -msse3 -mssse3)
else()
  add_compile_options(-march=armv8-a)
endif()

add_compile_options(-Wall -D_LINUX -D_REENTRANT -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas)
add_compile_options(-O3 -fno-signed-zeros -fno-math-errno -funroll-loops)

# -------- OpenCV（显式组件，避免只有 core 被链）--------
find_package(OpenCV REQUIRED
  COMPONENTS core imgproc imgcodecs highgui calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# -------- Eigen --------
find_package(Eigen3 REQUIRED)
if(NOT EIGEN3_INCLUDE_DIR)
  # 兼容一些旧版 FindEigen
  set(EIGEN3_INCLUDE_DIR ${EIGEN_INCLUDE_DIR})
endif()

# -------- Sophus（老版：非模板，提供 libSophus.so）--------
# 只需要头 + 库；旧 Sophus 通常装到 /usr/local
find_path(SOPHUS_INCLUDE_DIR NAMES sophus/se3.h sophus/so3.h
          PATHS /usr/local/include /usr/include)
find_library(SOPHUS_LIB NAMES Sophus sophus
             PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)

if(NOT SOPHUS_INCLUDE_DIR OR NOT SOPHUS_LIB)
  message(FATAL_ERROR "Old Sophus not found. Headers=${SOPHUS_INCLUDE_DIR}, Lib=${SOPHUS_LIB}\n"
                      "Make sure you installed the old (0.9.x) Sophus to /usr/local.")
endif()

# -------- ROS 可选 --------
set(USE_ROS TRUE)
if(USE_ROS)
  find_package(catkin REQUIRED COMPONENTS roscpp cmake_modules)
  catkin_package(
    DEPENDS Eigen3 OpenCV
    CATKIN_DEPENDS roscpp
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
  )
endif()

# 头文件
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SOPHUS_INCLUDE_DIR}
)

if(USE_ROS)
  include_directories(${catkin_INCLUDE_DIRS})
endif()

# 源文件
set(SOURCEFILES
  src/atan_camera.cpp
  src/omni_camera.cpp
  src/math_utils.cpp
  src/vision.cpp
  src/performance_monitor.cpp
  src/robust_cost.cpp
  src/user_input_thread.cpp
  src/pinhole_camera.cpp
  src/equidistant_camera.cpp
  src/polynomial_camera.cpp
  src/homography.cpp
  src/img_align.cpp
)

# 生成共享库
add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})

# 关键：链接顺序要把依赖库放在后面（GNU ld 需要）
# 同时加上 --no-as-needed 防止库被丢弃
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    -Wl,--no-as-needed
    ${OpenCV_LIBS}
    ${SOPHUS_LIB}
)

if(USE_ROS)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${catkin_LIBRARIES})
endif()

# ---- 测试可执行（再显式补链一次，保险）----
add_executable(test_vk_common_camera test/test_camera.cpp)
target_link_libraries(test_vk_common_camera
  PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS} ${SOPHUS_LIB})

add_executable(test_vk_common_triangulation test/test_triangulation.cpp)
target_link_libraries(test_vk_common_triangulation
  PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS} ${SOPHUS_LIB})

add_executable(test_vk_common_patch_score test/test_patch_score.cpp)
target_link_libraries(test_vk_common_patch_score
  PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS} ${SOPHUS_LIB})
