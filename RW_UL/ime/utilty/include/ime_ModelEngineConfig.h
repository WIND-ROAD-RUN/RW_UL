#pragma once

#include<string>
#include<vector>

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
	};

	enum class ModelType
	{
		yolov11_det,
		yolov11_seg
	};

	enum class ModelEngineDeployType
	{
		TensorRT,
		OnnxRuntime
	};
}