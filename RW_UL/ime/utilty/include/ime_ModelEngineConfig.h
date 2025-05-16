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
		//控制检测结果的最低置信度，过滤低置信度的候选框。
		float conf_threshold = 0.3f;
		//控制非极大值抑制的 IoU 阈值，移除重叠的候选框。
		float nms_threshold = 0.4f;
	public:
		std::vector<size_t> need_keep_classids{};
	public:
		std::string modelPath;
	public:
		ImagePretreatmentPolicy imagePretreatmentPolicy = ImagePretreatmentPolicy::LetterBox;
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