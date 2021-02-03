#pragma once
#include <iostream>
#include <stdio.h>
#include <cstdio>

#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"

#include "torch/torch.h"
#include <torch/script.h>


using namespace std;


/**  Struct to pass data to Unreal */
struct TransformData
{
	TransformData(float tx, float ty, float tz, float rfx, float rfy, float rfz, float rux, float ruy, float ruz) :
		tX(tx), tY(ty), tZ(tz), rfX(rfx), rfY(rfy), rfZ(rfz), ruX(rux), ruY(ruy), ruZ(ruz) {}
	float tX, tY, tZ;
	float rfX, rfY, rfZ;
	float ruX, ruY, ruZ;
};

/**  Class to manage whole pipeline, from camera capture to pose estimation */
class Estimator
{
public:

	/** Video feed */
	cv::VideoCapture _capture;

	/** File paths for models */
	const string face_detect_config_path = "C:/git-clone-tests/facial-pose-estimation-libtorch-dll/deploy.prototxt";
	const string face_detect_weight_path = "C:/git-clone-tests/facial-pose-estimation-libtorch-dll/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	const string blend_shape_detect_mdl_path = "C:/git-clone-tests/facial-pose-estimation-libtorch-dll/bs_detect.ptc";
	const string landmark_detect_mdl_path = "C:/git-clone-tests/facial-pose-estimation-libtorch-dll/landmark_detect.ptc";
	const string blend_shape_mdl_path = "C:/git-clone-tests/facial-pose-estimation-libtorch-dll/bs_model.ptc";

	/** Neural networks */
	cv::dnn::Net box_detector;
	torch::jit::script::Module blend_shape_detect_mdl;
	torch::jit::script::Module landmark_detect_mdl;
	torch::jit::script::Module blend_shape_mdl;

	/** Capture dimensions */
	int frame_width;
	int frame_height;
	int detect_ratio;

	/** Face detection res and line render res */
	int face_detect_res;
	int line_render_res;

	/**
	 * Storage for parameters from Unreal
	 */

	 /** Camera Zoom */
	float fov_zoom;

	/** Draw Axis on face */
	bool draw_points;

	/** Whether to animate eyes and nose in PnP sove */
	bool lock_eyes_nose;

	/** 
	 * Storage for reusable variables
	 */

	/** Stream image storage */
	cv::Mat frame;

	/** Blendshape prediction tensor */
	at::Tensor expression_tensor;

	/** 2d face landmark tensor */
	at::Tensor face_landmark_tensor;

	/** PnP landmark points */
	vector <cv::Point3d> predicted_points_3d;
	vector <cv::Point2d> landmark_points_2d;

	/** Output transforms for PnP solve */
	cv::Mat translation_vector = cv::Mat(cv::Size(3, 1), cv::DataType<double>::type);
	cv::Mat rotation_vector = cv::Mat(cv::Size(3, 1), cv::DataType<double>::type);

	/** Camera initiation */
	cv::Mat camera_matrix;
	cv::Mat dist_coeffs;

	/** 2d face box data */
	int face_width;
	int center_x;
	int center_y;

	/** ID correspondence map */
	const vector<int> point_ids = {
		6,7,8,9,10,17,18,19,
		20,21,22,23,24,25,26,
		27,28,29,30,31,32,33,
		34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,
		48,49,50,51,52,53,54,
		55,56,57,58,59,60,61,
		62,63,64,65,66,67 
	};

	/** IDs to be used for triangulating face pose */
	const vector<int> triangulation_ids = { 30,8, 36, 45, 48, 54 };

	/** Colors to use from drawing of line face */
	const vector<cv::Scalar> draw_colors = {
		cv::Scalar(0,0,0),cv::Scalar(0,0,128),
		cv::Scalar(128,0,0),cv::Scalar(128,128,256),
		cv::Scalar(0,256,256),cv::Scalar(256,0,256),
		cv::Scalar(256,256,128),cv::Scalar(128,0,256),
		cv::Scalar(128,128,128),cv::Scalar(128,0,128),
		cv::Scalar(128,256,0),cv::Scalar(256,256,256),
		cv::Scalar(256,128,0),cv::Scalar(0,128,128),
		cv::Scalar(0,0,256),cv::Scalar(256,128,128),
		cv::Scalar(0,256,128),cv::Scalar(128,128,0),
		cv::Scalar(256,128,256),cv::Scalar(0,128,256),
		cv::Scalar(128,256,256),cv::Scalar(256,0,0),
		cv::Scalar(256,256,0),cv::Scalar(0,128,0),
		cv::Scalar(0,256,0),cv::Scalar(256,0,128),
		cv::Scalar(128,256,128) };


public:

	Estimator();


	/**
	* Initiate OpenCV camera stream and Neural Networks.
	* @param out_camera_width - Width which OpenCV used for camera stream.
	* @param out_camera_height - Height which OpenCV used for camera stream.
	* @param cam_id - Which camera id OpenCV should try to use.
	* @param in_fov_zoom - Zoom amount for pinhole camera, to match Unreal.
	* @param draw - Wheher or not to draw technical indicators over frame.
	* @param lock_eyes_nose - Whether to lock eye and nose points for PnP solve.
	* @return Whether operation is succesful.
	*/
	int init(int& out_camera_width, int& out_camera_height, int in_detect_ratio, int cam_id, float in_fov_zoom, bool draw, bool lock_eyes_nose);

	/**
	* Close OpenCV connection to camera.
	*/
	void close();

	/**
	* Exectute whole facial pose estimation pipeline, and return result.
	* @param out_transform - Pointer where result transform is copied to.
	* @param out_expression - Pointer where result blendshapes are copied to.
	*/
	void detect(TransformData& out_transform, float* out_expression);

	/**
	* Use OpenCV neural net to detect face in frame.
	*/
	void bounding_box_detect();

	/**
	* Use Libtorch neural net to detect landmarks on cropped face.
	*/
	void landmark_detect();

	/**
	* Draw technical indicators on top of camera stream image (Usefull for debugging).
	*/
	void draw_solve();

	/**
	* Use Libtorch neural net to detect blendshape of face from line drawn image.
	* @param out_expression - Pointer where result blendshapes are copied to.
	*/
	void landmark_to_blendshapes(float* out_expression);

	/**
	* Perform PnP solve using OpenCV, using estimated face blendshaps and 2d landmarks.
	* @param out_transform - Pointer where result transformation is copied to.
	*/
	void pnp_solve(TransformData& out_transform);

	/**
	* Using 2d landmarks detected by neural net, draw line face representation.
	* @return OpenCV line drawing.
	*/
	cv::Mat get_line_face();

	/**
	* Get single frame from OpenCV camera stream, resize and reformat for Unreal.
	* @param data - Pointer to write OpenCV image to.
	* @param width - Resize width.
	* @param height - Resize height.
	* @return Whether operation is succesful.
	*/
	int get_raw_image_bytes(unsigned char* data, int width, int height);

	/**
	* Using blendshape values output from neural net, generate facial pose.
	* @return Estimatd keypoints to use for PnP solve.
	*/
	vector<cv::Point3d> get_pose();

	/**
	* Add single line to image using input coordinates.
	* @param image - OpenCV image to draw on.
	* @param x_points - Array of X facial landmark coordinates.
	* @param y_points - Array of Y facial landmark coordinates.
	* @param id_list - List of which landmarks to use to draw line.
	* @param color - Color of the line.
	*/
	static void build_line(cv::Mat* image, cv::Mat x_points, cv::Mat y_points, vector<int> id_list, cv::Scalar color);

};

