#include <ros/ros.h>
#include <tsdf_reconstruction_cuda/sdf_map.h>

// OpenCV specific includes
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace std;
using cv::cuda::GpuMat;

int *difffilter(cv::Mat src, cv::Mat dst);

class tsdf_reconstruction_cuda_Node
{
  public:
   tsdf_reconstruction_cuda_Node();
   virtual ~tsdf_reconstruction_cuda_Node();
   void subcribeTopics();
   void advertiseTopics();
   void depthCallback(const sensor_msgs::Image::ConstPtr& msg);
   void rgbCallback(const sensor_msgs::Image::ConstPtr& msg);

   cv::Mat depth_image, rgb_image;

   sdf_reconstruction::sdf_map *mySDF_map;
   sdf_reconstruction::GlobalConfiguration config;
   sdf_reconstruction::CameraParameters sensor_Param;

  private:
   ros::NodeHandle nh_;
   ros::NodeHandle nh_depth_img, nh_rgb_img;
   ros::Subscriber depth_img_sub, rgb_img_sub, cloud_sub;
   ros::Publisher cloud_registered;
};


tsdf_reconstruction_cuda_Node::tsdf_reconstruction_cuda_Node()
{
  nh_ = ros::NodeHandle("~");
  nh_.getParam("voxel_scale", config.voxel_scale);
  nh_.getParam("bfilter_kernel_size", config.bfilter_kernel_size);
  nh_.getParam("bfilter_color_sigma", config.bfilter_color_sigma);
  nh_.getParam("bfilter_spatial_sigma", config.bfilter_spatial_sigma);
  nh_.getParam("init_depth", config.init_depth);
  nh_.getParam("use_output_frame", config.use_output_frame);
  nh_.getParam("truncation_distance", config.truncation_distance);
  nh_.getParam("depth_cutoff_distance", config.depth_cutoff_distance);
  nh_.getParam("num_levels", config.num_levels);
  nh_.getParam("triangles_buffer_size", config.triangles_buffer_size);
  nh_.getParam("pointcloud_buffer_size", config.pointcloud_buffer_size);
  nh_.getParam("distance_threshold", config.distance_threshold);
  nh_.getParam("angle_threshold", config.angle_threshold);


  int volume_size_x, volume_size_y, volume_size_z;
  nh_.getParam("volume_size_x", volume_size_x);
  nh_.getParam("volume_size_y", volume_size_y);
  nh_.getParam("volume_size_z", volume_size_z);  
  config.volume_size = make_int3(volume_size_x, volume_size_y, volume_size_z);
  //std::cerr << "triangles_buffer_size: " << configuration.volume_size.x;

  nh_.getParam("icp_iterations_l0", config.icp_iterations[0]);
  nh_.getParam("icp_iterations_l1", config.icp_iterations[1]);
  nh_.getParam("icp_iterations_l2", config.icp_iterations[2]);
  //std::cerr << "icp_iterations: " << configuration.icp_iterations[0];

  nh_.getParam("image_width", sensor_Param.image_width);
  nh_.getParam("image_height", sensor_Param.image_height);
  nh_.getParam("focal_x", sensor_Param.focal_x);
  nh_.getParam("focal_y", sensor_Param.focal_y);
  nh_.getParam("principal_x", sensor_Param.principal_x);
  nh_.getParam("principal_y", sensor_Param.principal_y);

  mySDF_map = new sdf_reconstruction::sdf_map(sensor_Param, config);
}

tsdf_reconstruction_cuda_Node::~tsdf_reconstruction_cuda_Node()
{

}

void tsdf_reconstruction_cuda_Node::subcribeTopics()
{
  std::string subcribe_depth_topic= "/camera/depth/image"; 
  std::string subcribe_color_topic= "/camera/rgb/image_rect_color"; 
  rgb_img_sub = nh_rgb_img.subscribe (subcribe_color_topic, 1, &tsdf_reconstruction_cuda_Node::rgbCallback, this);  
  depth_img_sub = nh_depth_img.subscribe (subcribe_depth_topic, 1, &tsdf_reconstruction_cuda_Node::depthCallback, this);
  
}

void tsdf_reconstruction_cuda_Node::advertiseTopics()
{

}

void tsdf_reconstruction_cuda_Node::depthCallback (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;

  try
  {
    bridge = cv_bridge::toCvCopy(msg, "32FC1");
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform depth image.");
    return;
  }
  depth_image = bridge->image;
  cv::imshow("depth image", depth_image);
  cv::waitKey(3);
  if(!depth_image.empty())
  if(!rgb_image.empty())
  { 
     cv::imshow("RGB image", rgb_image);
     cv::waitKey(3); 
  }
  *mySDF_map->depthImg = bridge->image.clone();
  mySDF_map->pipeline();
}

void tsdf_reconstruction_cuda_Node::rgbCallback (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;
  try
  {
    bridge = cv_bridge::toCvCopy(msg, "bgr8");    
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform rgb image.");
    return;
  }

  rgb_image = bridge->image;
  Mat imageGray, out;
  cv::cvtColor(rgb_image, imageGray, CV_BGR2GRAY);
/*   GpuMat gpuImageGray(imageGray), gpuOut, bilateralOut;
  cv::cuda::bilateralFilter(gpuImageGray, bilateralOut, 7, 150, 150);
  cv::Mat reconvert(bilateralOut);
  imshow("out", reconvert);
  waitKey(3); */
  *mySDF_map->rgbImg = bridge->image.clone();

}

int main (int argc, char** argv)
{
  ros::init(argc, argv, "tsdf_reconstruction_cuda_node");
  tsdf_reconstruction_cuda_Node mainNode;
  mainNode.subcribeTopics();
  mainNode.advertiseTopics();
  ros::spin();
  return 0;
}

