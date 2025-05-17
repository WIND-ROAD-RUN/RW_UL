#include"imet_ModelEngineFactory_TensorRT.hpp"

#include"imet_ModelEngine_yolov11_det.hpp"
#include"imet_ModelEngine_yolov11_seg.hpp"
#include"imet_ModelEngine_yolov11_obb.hpp"

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {

	}
}logger;

namespace rw {
	namespace imet {
		static ModelEngine_Yolov11_det* createModelEngine_Yolov11_det(const ModelEngineConfig& config);
		static ModelEngine_Yolov11_seg* createModelEngine_Yolov11_seg(const ModelEngineConfig& config);
		static ModelEngine_Yolov11_obb* createModelEngine_Yolov11_obb(const ModelEngineConfig& config);


		std::unique_ptr<ModelEngine>
			ModelEngineFactory_TensorRT::createModelEngine
			(const ModelEngineConfig& config, ModelType modelType)
		{
			std::unique_ptr<ModelEngine> modelEngine = nullptr;
			switch (modelType)
			{
			case ModelType::yolov11_det:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_det(config));
			case ModelType::yolov11_seg:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_seg(config));
			case ModelType::yolov11_obb:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_obb(config));
			default:
				return nullptr;
			}
		}

		ModelEngine_Yolov11_det* createModelEngine_Yolov11_det(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_det* modelEngine = new ModelEngine_Yolov11_det(config.modelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}

			modelEngine->setConfig(config);
			return modelEngine;
		}

		ModelEngine_Yolov11_seg* createModelEngine_Yolov11_seg(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_seg* modelEngine = new ModelEngine_Yolov11_seg(config.modelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}
			modelEngine->setConfig(config);
			return modelEngine;
		}

		ModelEngine_Yolov11_obb* createModelEngine_Yolov11_obb(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_obb* modelEngine = new ModelEngine_Yolov11_obb(config.modelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}
			modelEngine->setConfig(config);
			return modelEngine;
		}

	}
}
