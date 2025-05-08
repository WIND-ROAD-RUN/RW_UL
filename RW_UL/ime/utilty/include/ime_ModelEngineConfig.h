#pragma once

#include<string>

namespace rw {
	struct ModelEngineConfig {
	public:
		//控制检测结果的最低置信度，过滤低置信度的候选框。
		float conf_threshold = 0.3f;
		//控制非极大值抑制的 IoU 阈值，移除重叠的候选框。
		float nms_threshold = 0.4f;
	public:
		std::string modelPath;
	};

	enum class ModelType
	{
		yolov11_obb,
		yolov11_seg
	};

	enum class ModelEngineDeployType
	{
		TensorRT,
		OnnxRuntime
	};
}