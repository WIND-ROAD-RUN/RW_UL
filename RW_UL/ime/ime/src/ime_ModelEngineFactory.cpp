#include"ime_ModelEngineFactory.h"

#include"imet_ModelEngineFactory_TensorRT.hpp"

namespace rw {
	std::unique_ptr<ModelEngine> ModelEngineFactory::createModelEngine(const ModelEngineConfig& config, ModelType modelType, ModelEngineDeployType deployType)
	{
		switch (deployType)
		{
		case rw::ModelEngineDeployType::TensorRT:
			return imet::ModelEngineFactory_TensorRT::createModelEngine(config, modelType);
		default:
			return nullptr;
		}
	}

}
