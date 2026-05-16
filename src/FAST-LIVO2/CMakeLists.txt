cmake_minimum_required(VERSION 2.8.3)
project(fast_livo)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
option(BUILD_CENTERPOINT "Build CenterPoint 3D detection node" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

# Specific settings for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Detect CPU architecture
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    # 32-bit ARM optimizations with NEON support
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) optimizations
#  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops") #-flto
#  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mtune=native -funroll-loops") #-flto
  message(STATUS "Using general x86 optimizations: -O3 -mtune=native -funroll-loops")
  add_definitions(-DX86_ARCH)
endif()

# Define project root directory
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Detect CPU core count for potential multithreading optimization
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

# Set the number of cores for multithreading
if(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
else()
  add_definitions(-DMP_PROC_NUM=1)
  message(STATUS "Single core detected. Multithreading disabled.")
endif()

# Check for OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
else()
  message(STATUS "OpenMP not found, proceeding without it")
endif()

# Check for mimalloc support
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

set(FAST_LIVO_CATKIN_COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  std_msgs
  pcl_ros
  tf
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

if(BUILD_CENTERPOINT)
  list(APPEND FAST_LIVO_CATKIN_COMPONENTS jsk_recognition_msgs)
endif()

# Find catkin and required dependencies
find_package(catkin REQUIRED COMPONENTS ${FAST_LIVO_CATKIN_COMPONENTS})

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread timer serialization)
find_package(OpenMP REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")
find_package (GeographicLib REQUIRED)

find_package(GTSAMCMakeTools)
find_package(GTSAM REQUIRED QUIET)

set(Sophus_LIBRARIES libSophus.so)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS ${FAST_LIVO_CATKIN_COMPONENTS}
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

# Include directories for dependencies
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GeographicLib_INCLUDE_DIRS}
  include
)

# Add libraries
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)
# Add the main executable
add_executable(fastlivo_mapping src/main.cpp)

# Link libraries to the executable
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${GeographicLib_LIBRARIES}
  gtsam
)

# Link mimalloc if found
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()

if(BUILD_CENTERPOINT)
  # -------- CenterPoint / TensorRT / CUDA / spconv -----------
  find_package(CUDA REQUIRED)

  if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
  else()
    set(TENSORRT_ROOT /opt/TensorRT)
  endif()

  if(DEFINED ENV{SPCONV_ROOT})
    set(SPCONV_ROOT $ENV{SPCONV_ROOT})
  else()
    set(SPCONV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libspconv)
  endif()

  set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
  set(TENSORRT_LIBRARY_DIRS ${TENSORRT_ROOT}/lib)

  if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  endif()
  set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

  set(arch ${CMAKE_SYSTEM_PROCESSOR})
  if(arch MATCHES "aarch64|ARM|arm64")
    set(SPCONV_LIB_DIR ${SPCONV_ROOT}/lib/aarch64)
  else()
    set(SPCONV_LIB_DIR ${SPCONV_ROOT}/lib/x86_64)
  endif()

  if(NOT EXISTS "${SPCONV_LIB_DIR}/libspconv.so")
    message(FATAL_ERROR "libspconv.so was not found. Set SPCONV_ROOT or install spconv before building with BUILD_CENTERPOINT=ON.")
  endif()

  include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${SPCONV_ROOT}/include
  )

  link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    ${SPCONV_LIB_DIR}
  )

  file(GLOB_RECURSE CENTERPOINT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/center_pointpillars/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/center_pointpillars/*.cpp)

  cuda_add_executable(centerpp_node
    ${CMAKE_CURRENT_SOURCE_DIR}/src/centerpp_node/centerpp_node.cpp
    ${CENTERPOINT_SOURCES}
  )

  target_compile_options(centerpp_node PRIVATE ${OpenMP_CXX_FLAGS})
  target_compile_definitions(centerpp_node PRIVATE PCL_NO_PRECOMPILE)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fopenmp")

  target_link_libraries(centerpp_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    gomp
    nvinfer
    nvinfer_plugin
    nvonnxparser
    spconv
  )

  target_include_directories(centerpp_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_dependencies(centerpp_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  install(TARGETS centerpp_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()
