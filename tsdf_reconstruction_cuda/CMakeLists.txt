cmake_minimum_required(VERSION 2.8.3)
project(tsdf_reconstruction_cuda)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED NO_MODULE PATHS /usr/local NO_DEFAULT_PATH)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_INCLUDE_DIRS)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(CUDA 8.0 REQUIRED)
#IF (CUDA_FOUND)
#    include_directories("${CUDA_INCLUDE_DIRS}")
    # Optional: Specify the arch of your CUDA hardware here
#    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52)
#ENDIF ()


catkin_package(
   INCLUDE_DIRS include
#  LIBRARIES tsdf_reconstruction_cuda
   CATKIN_DEPENDS cv_bridge pcl_conversions pcl_ros roscpp sensor_msgs std_msgs std_srvs visualization_msgs
   DEPENDS eigen
   LIBRARIES # TODO
#  DEPENDS system_lib
)

include_directories(include  ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
add_definitions(${EIGEN3_DEFINITIONS} "${PROJECT_SOURCE_DIR}/src/cuda/include")

set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
file(GLOB cuda_SRCS "${PROJECT_SOURCE_DIR}/src/cuda/*.cu")
set(LIB_TYPE STATIC)
cuda_add_library(cuda_sdf ${LIB_TYPE} "${cuda_SRCS}")

## Declare a C++ library
add_library(sdf_map
   src/sdf_map.cpp
)


## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/main.cpp src/sdf_map.cpp)

target_link_libraries(${PROJECT_NAME}_node ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS} sdf_map cuda_sdf)

