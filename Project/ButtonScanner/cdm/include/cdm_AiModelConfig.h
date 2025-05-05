#pragma once

#include <string>

namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}

namespace rw
{
	namespace cdm
	{
		enum class ModelType
		{
			Undefined = 0,//未定义
			Color = 1,//颜色
			BladeShape = 2//刀型
		};

		int ModelTypeToInt(ModelType type);
		ModelType IntToModelType(int type);

		class AiModelConfig
		{
		public:
			AiModelConfig() = default;
			~AiModelConfig() = default;

			AiModelConfig(const rw::oso::ObjectStoreAssembly& assembly);
			AiModelConfig(const AiModelConfig& buttonScannerMainWindow);

			AiModelConfig& operator=(const AiModelConfig& buttonScannerMainWindow);
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const AiModelConfig& account) const;
			bool operator!=(const AiModelConfig& account) const;
		public:
			ModelType modelType{ ModelType::Undefined };//模型类型
			std::string date{};//训练日期
			bool upLight{ false };
			bool downLight{ false };
			bool sideLight{ false };
			size_t exposureTime{ 100 };
			size_t gain{ 2 };
		public:
			std::string rootPath{};//模型数据相关根路径
			std::string name{};//模型名称
		public:
			long id{ -1 };//模型ID
		};
	}
}
