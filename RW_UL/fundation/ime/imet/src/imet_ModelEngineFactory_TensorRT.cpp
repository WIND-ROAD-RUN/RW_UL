#include"imet_ModelEngineFactory_TensorRT.hpp"

#include"imet_ModelEngine_yolov11_det.hpp"
#include "imet_ModelEngine_yolov11_det_cudaAcc.hpp"
#include"imet_ModelEngine_yolov11_seg.hpp"
#include"imet_ModelEngine_yolov11_obb.hpp"
#include "imet_ModelEngine_yolov11_seg_with_mask.hpp"

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
	}
}logger;

namespace rw {
	namespace imet {
		static ModelEngine_Yolov11_det* createModelEngine_Yolov11_det(const ModelEngineConfig& config);
		static ModelEngine_yolov11_det_cudaAcc* createModelEngine_Yolov11_det_cuda_acc(const ModelEngineConfig& config);
		static ModelEngine_Yolov11_seg* createModelEngine_Yolov11_seg(const ModelEngineConfig& config);
		static ModelEngine_Yolov11_obb* createModelEngine_Yolov11_obb(const ModelEngineConfig& config);
		static ModelEngine_Yolov11_seg_with_mask* createModelEngine_Yolov11_seg_with_mask(const ModelEngineConfig& config);

		std::unique_ptr<ModelEngine>
			ModelEngineFactory_TensorRT::createModelEngine
			(const ModelEngineConfig& config, ModelType modelType)
		{
			std::unique_ptr<ModelEngine> modelEngine = nullptr;
			switch (modelType)
			{
			case ModelType::Yolov11_Det:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_det(config));
			case ModelType::Yolov11_Det_Cuda_Acc:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_det_cuda_acc(config));
			case ModelType::Yolov11_Seg:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_seg(config));
			case ModelType::Yolov11_Obb:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_obb(config));
			case ModelType::Yolov11_Seg_with_mask:
				return std::unique_ptr<ModelEngine>(createModelEngine_Yolov11_seg_with_mask(config));
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

		ModelEngine_yolov11_det_cudaAcc* createModelEngine_Yolov11_det_cuda_acc(const ModelEngineConfig& config)
		{
			ModelEngine_yolov11_det_cudaAcc* modelEngine = new ModelEngine_yolov11_det_cudaAcc(config, logger);
			if (!modelEngine) {
				return nullptr;
			}
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

		ModelEngine_Yolov11_seg_with_mask* createModelEngine_Yolov11_seg_with_mask(const ModelEngineConfig& config)
		{
			ModelEngine_Yolov11_seg_with_mask* modelEngine = new ModelEngine_Yolov11_seg_with_mask(config.modelPath, logger);
			if (!modelEngine) {
				return nullptr;
			}
			modelEngine->setConfig(config);
			return modelEngine;
		}
	}
}