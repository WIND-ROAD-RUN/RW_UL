#include"imet_ModelEngineFactory_TensorRT.hpp"

#include"imet_ModelEngine_yolov11_obb.hpp"

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {

	}
}logger;

namespace rw {
	namespace imet {
		static ModelEngine_Yolov11_Obb* createModelEngine_Yolov11_Obb(const ModelEngineConfig& config);

		std::unique_ptr<ModelEngine>
			ModelEngineFactory_TensorRT::createModelEngine
			(const ModelEngineConfig& config, ModelType modelType)
		{
			std::unique_ptr<ModelEngine> modelEngine = nullptr;
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
			ModelEngine_Yolov11_Obb* modelEngine = new ModelEngine_Yolov11_Obb(config.ModelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}

			modelEngine->setConf_threshold(config.conf_threshold);
			modelEngine->setNms_threshold(config.nms_threshold);

			return modelEngine;
		}

	}
}
