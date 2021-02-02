#pragma once
#include "pose_estimate.h"
#include <iostream>
#include <stdio.h>
#include <cstdio>


Estimator PoseEstimator = Estimator();


extern "C" int __declspec(dllexport) __stdcall  Init(int& outCameraWidth, int& outCameraHeight, int detectRatio, int camId, float fovZoom, bool draw, bool lockEyesNose)
{
	return PoseEstimator.init(outCameraWidth, outCameraHeight, detectRatio, camId, fovZoom, draw, lockEyesNose);
}


extern "C" void __declspec(dllexport) __stdcall  Close()
{
	PoseEstimator.close();
}


extern "C" void __declspec(dllexport) __stdcall Detect(TransformData& outFaces, float* outExpression)
{
	PoseEstimator.detect(outFaces, outExpression);
}


extern "C" int __declspec(dllexport) __stdcall GetRawImageBytes(unsigned char* data, int width, int height)
{
	return PoseEstimator.get_raw_image_bytes(data, width, height);
}
