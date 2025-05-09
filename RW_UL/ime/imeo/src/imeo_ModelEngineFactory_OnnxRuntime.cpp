#include "imeo_ModelEngineFactory_OnnxRuntime.hpp"

#include"imeo_ModelEngine_yolov11_obb.hpp"

#include<memory>

namespace rw
{
	namespace imeo
	{
		static ModelEngine_Yolov11_Obb* createModelEngine_Yolov11_Obb(const ModelEngineConfig& config);

		std::unique_ptr<ModelEngine> ModelEngineFactory_OnnxRuntime::createModelEngine(const ModelEngineConfig& config, ModelType modelType)
		{
			switch (modelType)
			{
			case ModelType::yolov11_obb:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_Obb(config));
			default:
				return nullptr;
			}
		}

		ModelEngine_Yolov11_Obb* createModelEngine_Yolov11_Obb(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_Obb* modelEngine = new ModelEngine_Yolov11_Obb(config.modelPath);
			if (!modelEngine) {
				return nullptr;
			}
			modelEngine->setConf_threshold(config.conf_threshold);
			modelEngine->setNms_threshold(config.nms_threshold);

			return modelEngine;
		}
	}
}
