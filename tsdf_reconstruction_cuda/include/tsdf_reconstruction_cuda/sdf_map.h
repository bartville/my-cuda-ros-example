#ifndef SDF_MAP_H
#define SDF_MAP_H

#include <iostream>
#include <tsdf_reconstruction_cuda/data_types.h>

// OpenCV specific includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace std;
using cv::cuda::GpuMat;

namespace sdf_reconstruction 
{

    class sdf_map
    {
        public:
            sdf_map(const CameraParameters _camera_parameters,
                    const GlobalConfiguration _configuration);
            virtual ~sdf_map();
            void pre_processing();
            void pipeline();
            FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                      const CameraParameters& camera_params,
                                      const size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, const float spatial_sigma);


            cv::Mat *depthImg, *rgbImg;

        private:
            // Internal parameters, not to be changed after instantiation
            const CameraParameters camera_parameters;
            const GlobalConfiguration configuration;
        
            // The global volume (containing tsdf and color)
            VolumeData volume;

            // The model data for the current frame
            ModelData model_data;

            // Poses: Current and all previous
            Eigen::Matrix4f current_pose;
            std::vector<Eigen::Matrix4f> poses;

            // Frame ID and raycast result for output purposes
            size_t frame_id;
            cv::Mat last_model_frame;
    };
}
#endif // SDF_MAP_H