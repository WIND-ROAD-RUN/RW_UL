#pragma once

#include"ime_ModelEngine.h"
#include"ime_ModelEngineConfig.h"
#include<memory>

namespace rw {
	namespace imet {
		enum class TensorRTModelType
		{
			yolov11_obb,
			yolov11_seg
		};

		class ModelEngineFactory
		{
		public:
			static std::unique_ptr<ModelEngine> createModelEngine(const ModelEngineConfig& config, TensorRTModelType modelType);
		};
	}
}