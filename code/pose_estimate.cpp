#pragma once

#include <iostream>
#include <stdio.h>
#include <cstdio>
#include "torch/torch.h"
#include <torch/script.h>

#include "pose_estimate.h"


Estimator::Estimator()
{
	// Load networks
	box_detector = cv::dnn::readNetFromCaffe(face_detect_config_path, face_detect_weight_path);
	box_detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// Set resolution
	frame_width = 1920;
	frame_height = 1080;

	// Face detection res
	face_detect_res = 128;
	line_render_res = 128;
}


int Estimator::init(int& out_camera_width, int& out_camera_height, int in_detect_ratio, int cam_id, float in_fov_zoom, bool draw, bool lock_eyes_nose)
{
	// Set parameters from Unreal
	draw_points = draw;
	detect_ratio = in_detect_ratio;
	fov_zoom = in_fov_zoom;

	// Init neural nets
	blend_shape_detect_mdl = torch::jit::load(blend_shape_detect_mdl_path, torch::kCUDA);
	landmark_detect_mdl = torch::jit::load(landmark_detect_mdl_path, torch::kCUDA);
	blend_shape_mdl = torch::jit::load(blend_shape_mdl_path, torch::kCUDA);
	blend_shape_mdl.setattr("lock_eyes_nose", lock_eyes_nose);

	// Init video feed
	_capture = cv::VideoCapture(cam_id);

	if (!_capture.isOpened())
		return -2;

	// Set resolution
	_capture.set(cv::CAP_PROP_FRAME_WIDTH, out_camera_width);
	_capture.set(cv::CAP_PROP_FRAME_HEIGHT, out_camera_height);

	// Get actual result resolution
	out_camera_width = _capture.get(cv::CAP_PROP_FRAME_WIDTH);
	out_camera_height = _capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	frame_width = out_camera_width;
	frame_height = out_camera_height;

	return 0;
}


void Estimator::close()
{
	_capture.release();
}


void Estimator::detect(TransformData& out_transform, float* out_expression)
{
	_capture >> frame;
	if (frame.empty())
		return;

	bounding_box_detect();

	landmark_detect();

	landmark_to_blendshapes(out_expression);

	pnp_solve(out_transform);

	if (draw_points)
	{
		// Draw points and axis on face
		draw_solve();
	}
}


int Estimator::get_raw_image_bytes(unsigned char* data, int width, int height)
{
	if (frame.empty())
		return 10;

	// Resize to output texture res
	cv::Mat tex_mat(height, width, frame.type());
	cv::resize(frame,
		tex_mat,
		tex_mat.size(),
		cv::INTER_CUBIC);

	//Convert from RGB to ARGB 
	cv::Mat argb_img;
	cv::cvtColor(tex_mat,
		argb_img,
		cv::COLOR_BGR2BGRA);

	vector<cv::Mat> bgra;
	cv::split(argb_img, bgra);
	cv::swap(bgra[0], bgra[3]);
	cv::swap(bgra[1], bgra[2]);

	// Copy data back to pointer
	std::memcpy(data,
		argb_img.data,
		argb_img.total() * argb_img.elemSize());

	return 0;
}


void Estimator::draw_solve()
{
	// Build axis from nose
	vector<cv::Point3d> pose_axis_3d;
	vector<cv::Point2d> pose_axis_2d;
	pose_axis_3d.push_back(predicted_points_3d[0] + cv::Point3d(0, 0, 400.0));
	pose_axis_3d.push_back(predicted_points_3d[0] + cv::Point3d(400, 0, 0));
	pose_axis_3d.push_back(predicted_points_3d[0] + cv::Point3d(0, 400, 0));

	// Project points
	vector<cv::Point2d> predicted_face_2d;

	cv::projectPoints(predicted_points_3d,
		rotation_vector,
		translation_vector,
		camera_matrix,
		dist_coeffs,
		predicted_face_2d);

	cv::projectPoints(pose_axis_3d, 
		rotation_vector,
		translation_vector,
		camera_matrix,
		dist_coeffs,
		pose_axis_2d);

	// Draw points
	for (int i = 0; i < 6; i++)
	{
		cv::circle(frame,
			cv::Point(landmark_points_2d[i].x,
			landmark_points_2d[i].y),
			4,
			cv::Scalar(255, 0, 0),
			3);

		cv::circle(frame,
			cv::Point(predicted_face_2d[i].x,
			predicted_face_2d[i].y),
			4,
			cv::Scalar(0, 0, 255),
			3);
	}

	// Draw axis lines
	cv::line(frame,
		predicted_face_2d[0],
		pose_axis_2d[0],
		cv::Scalar(255, 0, 0),
		2);

	cv::line(frame,
		predicted_face_2d[0],
		pose_axis_2d[1],
		cv::Scalar(0, 255, 0),
		2);

	cv::line(frame, 
		predicted_face_2d[0],
		pose_axis_2d[2],
		cv::Scalar(0, 0, 255),
		2);
}


void Estimator::pnp_solve(TransformData& out_transform)
{
	// Retrieve facial points with blenshapes applied
	predicted_points_3d = get_pose();

	// Prepair face points for perspective solve
	landmark_points_2d.clear();
	for (int id : triangulation_ids)
	{
		cv::Point2d point_2d = cv::Point2d(face_landmark_tensor[0][id][0].item<float>()* detect_ratio,
			face_landmark_tensor[0][id][1].item<float>()* detect_ratio);

		landmark_points_2d.push_back(point_2d);
	}

	// Generate fake camera matrix
	double focal_length = frame.cols*fov_zoom;
	cv::Point2d center = cv::Point2d(frame.cols / 2, frame.rows / 2);
	camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

	// Output rotation and translation, defaulting to in front of the camera
	translation_vector.at< double>(0) = 0;
	translation_vector.at< double>(1) = 0;
	translation_vector.at< double>(2) = 3200;

	rotation_vector.at< double>(0) = -3.2f;
	rotation_vector.at< double>(1) = 0.0f;
	rotation_vector.at< double>(2) = 0.0f;
	cv::Mat rot_mat;

	// Solve for pose
	cv::solvePnP(predicted_points_3d,
		landmark_points_2d,
		camera_matrix,
		dist_coeffs,
		rotation_vector,
		translation_vector,
		true,
		cv::SOLVEPNP_ITERATIVE);

	// Convert rotation to Matrix
	cv::Rodrigues(rotation_vector, rot_mat);

	// Export transform
	out_transform = TransformData(translation_vector.at<double>(0),
		translation_vector.at<double>(1),
		translation_vector.at<double>(2),
		rot_mat.at<double>(2, 0),
		rot_mat.at<double>(2, 1),
		rot_mat.at<double>(2, 2),
		rot_mat.at<double>(1, 0),
		rot_mat.at<double>(1, 1),
		rot_mat.at<double>(1, 2));
}


void Estimator::landmark_to_blendshapes(float* out_expression)
{
	// Construct line image for expression detection
	cv::Mat bs_mat = get_line_face();
	cv::Mat bs_mat_flipped, bs_mat_32;
	cv::flip(bs_mat, bs_mat_flipped, 1);

	// Make 32 bit float
	bs_mat_flipped.convertTo(bs_mat_32, CV_32FC3);

	// Convert to Torch tensor
	at::Tensor input_tensor = torch::from_blob(bs_mat_32.data,
		{ 1, line_render_res, line_render_res, 3 });

	// Normalize color
	input_tensor = (input_tensor / 127.5)-1;
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
	input_tensor = input_tensor.to(torch::kCUDA);

	// Pass data through neural net to get blendshape values
	torch::NoGradGuard no_grad_;
	c10::IValue output = blend_shape_detect_mdl.run_method("forward", input_tensor);
	expression_tensor = output.toTensor().clamp(0,1);
	expression_tensor = expression_tensor.detach();

	auto expression_copy = expression_tensor.to(torch::kCPU);

	// Copy tensor data into pointer
	std::memcpy(out_expression,
		expression_copy.contiguous().data<float>(),
		sizeof(float) *expression_copy.numel());
}


void Estimator::landmark_detect()
{
	// Run landmark detection
	cv::Mat half_frame(frame_height / detect_ratio,
		frame_width / detect_ratio,
		frame.type());

	cv::resize(frame,
		half_frame,
		half_frame.size(),
		cv::INTER_CUBIC);

	// Crop image
	cv::Rect crop_region(center_x-face_width,
		center_y-face_width,
		face_width*2,
		face_width*2);

	cv::Mat frame_crop = frame(crop_region);
	cv::Mat frame_120(256,256, frame_crop.type());

	cv::resize(frame_crop,
		frame_120,
		frame_120.size(),
		cv::INTER_CUBIC);

	// Convert to RGB
	cv::cvtColor(frame_120, frame_120, cv::COLOR_BGR2RGB);

	// Make 32 bit float
	cv::Mat frame_120_32;
	frame_120.convertTo(frame_120_32, CV_32FC3);

	// Convert to Torch tensor and normalize
	auto input_tensor = torch::from_blob(frame_120_32.data,
		{ 1,256,256, 3 });

	// Normalize color
	auto mean = torch::tensor({ 0.485, 0.456, 0.406 });
	auto std = torch::tensor({ 0.229, 0.224, 0.225 });
	
	input_tensor = (input_tensor / 255.0 - mean) / std;
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
	input_tensor = input_tensor.to(torch::kCUDA);

	// Pass image data through neural net to landmark coords
	torch::NoGradGuard no_grad_;
	c10::IValue output = landmark_detect_mdl.run_method("forward", input_tensor);
	face_landmark_tensor = output.toTensor();
	face_landmark_tensor = face_landmark_tensor.detach().to(torch::kCPU);

	// Normalize pixel coordinates
	float scale_mult = ((face_width * 2.0) / 256.0);
	face_landmark_tensor *= scale_mult ;
	face_landmark_tensor += torch::tensor({ center_x - (face_width), center_y - (face_width)});
}


void Estimator::bounding_box_detect()
{
	// Convert frame to blob, and drop into Face Box Detector Netowrk
	cv::Mat blob, out;
	blob = cv::dnn::blobFromImage(frame,
		1.0,
		cv::Size(300, 300),
		(104, 117, 123),
		false,
		false);

	box_detector.setInput(blob);
	cv::Mat detection = box_detector.forward();

	cv::Mat detectionMat(detection.size[2],
		detection.size[3],
		CV_32F,
		detection.ptr<float>());

	// Check results and only take the most confident prediction
	float largest_conf = 0;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > .5)
		{
			// Get dimensions
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * (frame_width / detect_ratio));
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * (frame_height / detect_ratio));
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * (frame_width / detect_ratio));
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * (frame_height / detect_ratio));

			// Generate square dimensions
			int temp_face_width = max(x2 - x1, y2 - y1) / 2;
			int temp_center_x = ((x2 + x1) / 2);
			int temp_center_y = ((y2 + y1) / 2);

			// Store most accurate result
			if (confidence > largest_conf)
			{
				face_width = temp_face_width;
				center_x = temp_center_x;
				center_y = temp_center_y;
				largest_conf = confidence;
			}
		}
	}
}


std::vector<cv::Point3d>  Estimator::get_pose()
{
	// Generate point positions from blendshape values
	torch::NoGradGuard no_grad_;
	auto output = blend_shape_mdl.run_method("forward", expression_tensor[0]);
	auto points = output.toTensor();
	points = points * 5000.f;
	auto points_cpu = points.detach().to(torch::kCPU);

	// Save output points to array for opencv
	std::vector<cv::Point3d> out_points;
	out_points.clear();
	for (int i = 0; i < 6; i++)
	{
		cv::Point3d new_point(points_cpu[i][0].clone().item<float>(),
			points_cpu[i][1].clone().item<float>(),
			points_cpu[i][2].clone().item<float>());

		out_points.push_back(new_point);
	}
	return out_points;
}


void  Estimator::build_line(cv::Mat* image, cv::Mat x_points, cv::Mat y_points, vector<int> id_list, cv::Scalar color)
{

	vector<cv::Point> new_point_list;

	cv::Mat line_image(image->rows,
		image->cols,
		CV_8UC3);

	line_image = 0;

	// Loop through points to build list
	for (int id : id_list)
	{
		new_point_list.push_back(
			cv::Point(x_points.at<float>(id, 0), y_points.at<float>(id, 0))
		);
	}

	// Build line from list
	cv::polylines(line_image,
		new_point_list,
		false, 
		color,
		1, 
		cv::LINE_AA,
		0);

	// Use max color, instead of directl overlay
	cv::max(line_image,
		*image, 
		*image);
}


cv::Mat Estimator::get_line_face()
{
	// Draw line image from Face Landmarks
	int point_size = point_ids.size();
	cv::Mat x_points(point_size, 1, CV_32FC1);
	cv::Mat y_points(point_size, 1, CV_32FC1);

	int count = 0;
	for (int id : point_ids)
	{
		x_points.at<float>(count, 0) = face_landmark_tensor[0][id][0].clone().item<float>();
		y_points.at<float>(count, 0) = face_landmark_tensor[0][id][1].clone().item<float>();
		count++;
	}
	cv::Mat bs_mat = cv::Mat::zeros(line_render_res, line_render_res, CV_8UC3);
	bs_mat = 0;

	// Measure value range, and then center values
	double min_val, max_val;
	cv::minMaxLoc(x_points, &min_val, &max_val);
	double width = max_val - min_val;
	double mean_x = (max_val + min_val) / 2;
	cv::minMaxLoc(y_points, &min_val, &max_val);
	double height = max_val - min_val;
	double mean_y = (max_val + min_val) / 2;

	x_points = x_points - mean_x;
	y_points = y_points - mean_y;
	x_points = (x_points / std::max(width, height))*(line_render_res * .9);
	y_points = (y_points / std::max(width, height))*(line_render_res * .9);
	x_points = x_points + (line_render_res / 2);
	y_points = y_points + (line_render_res / 2);

	vector<cv::Scalar>temp_colors(draw_colors);

	// Chin
	build_line(&bs_mat,
		x_points,
		y_points,
		vector<int>{0, 1, 2, 3, 4},
		temp_colors.back());

	temp_colors.pop_back();

	// Right Eye Broww
	build_line(&bs_mat,
		x_points,
		y_points,
		vector<int>{5, 6, 7, 8, 9},
		temp_colors.back());

	temp_colors.pop_back();

	// Left Eye Brow
	build_line(&bs_mat,
		x_points,
		y_points,
		vector<int>{10, 11, 12, 13, 14},
		temp_colors.back());

	temp_colors.pop_back();

	// Eyes
	build_line(&bs_mat,
		x_points,
		y_points,
		vector<int>{24, 25, 26, 27}, 
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat, 
		x_points, 
		y_points, 
		vector<int>{27, 28, 29, 24},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat, 
		x_points, 
		y_points, 
		vector<int>{30, 31, 32, 33}, 
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points,
		y_points,
		vector<int>{33, 34, 35, 30}, 
		temp_colors.back());

	temp_colors.pop_back();

	// Outer Mouth
	build_line(&bs_mat, 
		x_points, 
		y_points,
		vector<int>{36, 37, 38, 39},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points,
		y_points, 
		vector<int>{39, 40, 41, 42}, 
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points,
		y_points,
		vector<int>{42, 43, 44, 45},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points, 
		y_points,
		vector<int>{45, 46, 47, 36},
		temp_colors.back());

	temp_colors.pop_back();

	// Inner Mouth
	build_line(&bs_mat,
		x_points,
		y_points, 
		vector<int>{48, 49, 50}, 
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points, 
		y_points, 
		vector<int>{50, 51, 52}, 
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points, 
		y_points, 
		vector<int>{52, 53, 54}, 
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points,
		y_points, 
		vector<int>{54, 55, 48},
		temp_colors.back());

	temp_colors.pop_back();

	// Nose
	build_line(&bs_mat,
		x_points, 
		y_points,
		vector<int>{15, 16, 17, 18},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points,
		y_points, 
		vector<int>{18, 19},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points, 
		y_points, 
		vector<int>{19, 20, 21},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points,
		y_points, 
		vector<int>{21, 22, 23},
		temp_colors.back());

	temp_colors.pop_back();

	build_line(&bs_mat,
		x_points, 
		y_points,
		vector<int>{18, 23},
		temp_colors.back());

	temp_colors.pop_back();

	return bs_mat;
}

