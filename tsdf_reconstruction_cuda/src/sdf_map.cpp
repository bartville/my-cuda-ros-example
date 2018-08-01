#include <tsdf_reconstruction_cuda/sdf_map.h>

//int *difffilter(cv::Mat src, cv::Mat dst);

namespace sdf_reconstruction 
{
    namespace cuda { // Forward declare CUDA functions
            /* void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map); */
            int *difffilter(cv::Mat src, cv::Mat dst);
            
        }
        
    sdf_map::sdf_map(const CameraParameters _camera_parameters,
                    const GlobalConfiguration _configuration) :
                    camera_parameters(_camera_parameters), configuration(_configuration),
                    volume(_configuration.volume_size, _configuration.voxel_scale),
                    model_data(_configuration.num_levels, _camera_parameters),
                    current_pose{}, poses{}, frame_id{0}, last_model_frame{}
    {
        // The pose starts in the middle of the cube, offset along z by the initial depth
        current_pose.setIdentity();
        current_pose(0, 3) = _configuration.volume_size.x / 2 * _configuration.voxel_scale;
        current_pose(1, 3) = _configuration.volume_size.y / 2 * _configuration.voxel_scale;
        current_pose(2, 3) = _configuration.volume_size.z / 2 * _configuration.voxel_scale - _configuration.init_depth;

        depthImg = new cv::Mat(480, 640, CV_32FC1, Scalar(0));
        rgbImg = new cv::Mat(480, 640, CV_8UC3, Scalar(0)); 
    }

    sdf_map::~sdf_map()
    {
    }

    void sdf_map::pipeline()
    {
        //pre_processing();
        // STEP 1: Surface measurement
        FrameData frame_data = surface_measurement(*depthImg, camera_parameters,
                                configuration.num_levels,
                                configuration.depth_cutoff_distance,
                                configuration.bfilter_kernel_size,
                                configuration.bfilter_color_sigma,
                                configuration.bfilter_spatial_sigma);

    }

    void sdf_map::pre_processing()
    {
        //std::cerr << "current_pose: " << current_pose << "\n";
        Mat out;
        out = cv::Mat::zeros(depthImg->size(), CV_8UC1);
        cuda::difffilter(*depthImg, out);
        imshow("sobel", out);
        waitKey(3);

    }

    FrameData sdf_map::surface_measurement(const cv::Mat_<float>& input_frame,
                                      const CameraParameters& camera_params,
                                      const size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, const float spatial_sigma)
        {
            // Initialize frame data
            FrameData data(num_levels);

            // Allocate GPU memory
            for (size_t level = 0; level < num_levels; ++level) {
                const int width = camera_params.level(level).image_width;
                const int height = camera_params.level(level).image_height;

                data.depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
                data.smoothed_depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);

                data.color_pyramid[level] = cv::cuda::createContinuous(height, width, CV_8UC3);

                data.vertex_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
                data.normal_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
            }

            // Start by uploading original frame to GPU
            data.depth_pyramid[0].upload(input_frame);

            // Build pyramids and filter bilaterally on GPU
            cv::cuda::Stream stream;
            for (size_t level = 1; level < num_levels; ++level)
                cv::cuda::pyrDown(data.depth_pyramid[level - 1], data.depth_pyramid[level], stream);
            for (size_t level = 0; level < num_levels; ++level) {
                cv::cuda::bilateralFilter(data.depth_pyramid[level], // source
                                          data.smoothed_depth_pyramid[level], // destination
                                          kernel_size,
                                          color_sigma,
                                          spatial_sigma,
                                          cv::BORDER_DEFAULT,
                                          stream);
            }
            stream.waitForCompletion();

            // Compute vertex and normal maps
            for (size_t level = 0; level < num_levels; ++level) {
                /* cuda::compute_vertex_map(data.smoothed_depth_pyramid[level], data.vertex_pyramid[level],
                                         depth_cutoff, camera_params.level(level));
                cuda::compute_normal_map(data.vertex_pyramid[level], data.normal_pyramid[level]); */

                Mat out;
                out = cv::Mat::zeros(depthImg->size(), CV_8UC1);
                cuda::difffilter(*depthImg, out);
                imshow("sobel", out);
                waitKey(3);
            } 

            return data;
        }

}