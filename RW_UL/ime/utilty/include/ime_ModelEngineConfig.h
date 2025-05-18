#pragma once

#include<string>
#include<vector>

#include"opencv2/opencv.hpp"

namespace rw {
	enum class ImagePretreatmentPolicy
	{
		Resize = 0,
		LetterBox = 1,
		CenterCrop = 2,
		AdaptiveResize = 3
	};

	struct ModelEngineConfig {
	public:
		//Confidence threshold to filter out low-confidence candidates.
		float conf_threshold = 0.3f;
		//Control the maximum number of candidates to be retained.
		float nms_threshold = 0.4f;
	public:
		//The set of classid will be together for nms
		std::vector<size_t> classids_nms_together{};
	public:
		//The path of the model engine file.
		std::string modelPath;
	public:
		//The type of model engine.
		ImagePretreatmentPolicy imagePretreatmentPolicy = ImagePretreatmentPolicy::Resize;
		cv::Scalar letterBoxColor{ 0, 0, 0 };
		cv::Scalar centerCropColor{ 0, 0, 0 };
	};

	enum class ModelType
	{
		yolov11_det,
		yolov11_seg,
	    yolov11_obb
	};

	enum class ModelEngineDeployType
	{
		TensorRT,
		OnnxRuntime
	};
}