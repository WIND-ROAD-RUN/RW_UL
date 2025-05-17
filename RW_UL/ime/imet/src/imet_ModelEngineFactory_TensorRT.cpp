#include"imet_ModelEngineFactory_TensorRT.hpp"

#include"imet_ModelEngine_yolov11_det.hpp"
#include"imet_ModelEngine_yolov11_seg.hpp"

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {

	}
}logger;

namespace rw {
	namespace imet {
		static ModelEngine_Yolov11_det* createModelEngine_Yolov11_Obb(const ModelEngineConfig& config);
		static ModelEngine_Yolov11_seg* createModelEngine_Yolov11_seg(const ModelEngineConfig& config);


		std::unique_ptr<ModelEngine>
			ModelEngineFactory_TensorRT::createModelEngine
			(const ModelEngineConfig& config, ModelType modelType)
		{
			std::unique_ptr<ModelEngine> modelEngine = nullptr;
			switch (modelType)
			{
			case ModelType::yolov11_det:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_Obb(config));
			case ModelType::yolov11_seg:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_seg(config));
			default:
				return nullptr;
			}
		}

		ModelEngine_Yolov11_det* createModelEngine_Yolov11_Obb(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_det* modelEngine = new ModelEngine_Yolov11_det(config.modelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}

			modelEngine->setConf_threshold(config.conf_threshold);
			modelEngine->setNms_threshold(config.nms_threshold);

			return modelEngine;
		}

		ModelEngine_Yolov11_seg* createModelEngine_Yolov11_seg(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_seg* modelEngine = new ModelEngine_Yolov11_seg(config.modelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}
			modelEngine->setConf_threshold(config.conf_threshold);
			modelEngine->setNms_threshold(config.nms_threshold);
			modelEngine->setNeed_keep_classids(config.classids_nms_together);
			return modelEngine;
		}

	}
}
