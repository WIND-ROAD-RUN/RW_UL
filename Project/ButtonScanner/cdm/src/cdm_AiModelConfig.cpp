#include"cdm_AiModelConfig.h"

#include"oso_core.h"

namespace rw
{
	namespace cdm
	{
		int ModelTypeToInt(ModelType type)
		{
			switch (type)
			{
			case ModelType::Undefined:
				return 0;
			case ModelType::Color:
				return 1;
			case ModelType::BladeShape:
				return 2;
			default:
				return -1;
			}
		}

		ModelType IntToModelType(int type)
		{
			switch (type)
			{
			case 0:
				return ModelType::Undefined;
			case 1:
				return ModelType::Color;
			case 2:
				return ModelType::BladeShape;
			default:
				return ModelType::Undefined;
			}
		}

		AiModelConfig::AiModelConfig(const rw::oso::ObjectStoreAssembly& assembly)
		{
			auto isAccountAssembly = assembly.getName();
			if (isAccountAssembly != "$class$AiModelConfig$") {
				throw std::runtime_error("Assembly is not $class$AiModelConfig$");
			}
			auto modelTypeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$modelType$"));
			if (!modelTypeItem) {
				throw std::runtime_error("$variable$modelType is not found");
			}
			modelType = IntToModelType(modelTypeItem->getValueAsInt());
			auto dateItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$date$"));
			if (!dateItem) {
				throw std::runtime_error("$variable$date is not found");
			}
			date = dateItem->getValueAsString();
			auto upLightItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$upLight$"));
			if (!upLightItem) {
				throw std::runtime_error("$variable$upLight is not found");
			}
			upLight = upLightItem->getValueAsBool();
			auto downLightItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$downLight$"));
			if (!downLightItem) {
				throw std::runtime_error("$variable$downLight is not found");
			}
			downLight = downLightItem->getValueAsBool();
			auto sideLightItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$sideLight$"));
			if (!sideLightItem) {
				throw std::runtime_error("$variable$sideLight is not found");
			}
			sideLight = sideLightItem->getValueAsBool();
			auto exposureTimeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$exposureTime$"));
			if (!exposureTimeItem) {
				throw std::runtime_error("$variable$exposureTime is not found");
			}
			exposureTime = exposureTimeItem->getValueAsInt();
			auto gainItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$gain$"));
			if (!gainItem) {
				throw std::runtime_error("$variable$gain is not found");
			}
			gain = gainItem->getValueAsInt();

			auto rootPathItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$rootPath$"));
			if (!rootPathItem) {
				throw std::runtime_error("$variable$rootPath is not found");
			}
			rootPath = rootPathItem->getValueAsString();

			auto nameItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$name$"));
			if (!nameItem) {
				throw std::runtime_error("$variable$name is not found");
			}
			name = nameItem->getValueAsString();

			auto idItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$id$"));
			if (!idItem) {
				throw std::runtime_error("$variable$id is not found");
			}
			id = idItem->getValueAsLong();
		}

		AiModelConfig::AiModelConfig(const AiModelConfig& buttonScannerMainWindow)
		{
			modelType = buttonScannerMainWindow.modelType;
			date = buttonScannerMainWindow.date;
			upLight = buttonScannerMainWindow.upLight;
			downLight = buttonScannerMainWindow.downLight;
			sideLight = buttonScannerMainWindow.sideLight;
			exposureTime = buttonScannerMainWindow.exposureTime;
			gain = buttonScannerMainWindow.gain;
			rootPath = buttonScannerMainWindow.rootPath;
			name = buttonScannerMainWindow.name;
			id = buttonScannerMainWindow.id;
		}

		AiModelConfig& AiModelConfig::operator=(const AiModelConfig& buttonScannerMainWindow)
		{
			if (this != &buttonScannerMainWindow) {
				modelType = buttonScannerMainWindow.modelType;
				date = buttonScannerMainWindow.date;
				upLight = buttonScannerMainWindow.upLight;
				downLight = buttonScannerMainWindow.downLight;
				sideLight = buttonScannerMainWindow.sideLight;
				exposureTime = buttonScannerMainWindow.exposureTime;
				gain = buttonScannerMainWindow.gain;
				rootPath = buttonScannerMainWindow.rootPath;
				name = buttonScannerMainWindow.name;
				id = buttonScannerMainWindow.id;
			}
			return *this;
		}

		AiModelConfig::operator oso::ObjectStoreAssembly() const
		{
			rw::oso::ObjectStoreAssembly assembly;
			assembly.setName("$class$AiModelConfig$");

			auto modelTypeItem = std::make_shared<oso::ObjectStoreItem>();
			modelTypeItem->setName("$variable$modelType$");
			modelTypeItem->setValueFromInt(ModelTypeToInt(modelType));
			assembly.addItem(modelTypeItem);

			auto dateItem = std::make_shared<oso::ObjectStoreItem>();
			dateItem->setName("$variable$date$");
			dateItem->setValueFromString(date);
			assembly.addItem(dateItem);

			auto upLightItem = std::make_shared<oso::ObjectStoreItem>();
			upLightItem->setName("$variable$upLight$");
			upLightItem->setValueFromBool(upLight);
			assembly.addItem(upLightItem);

			auto downLightItem = std::make_shared<oso::ObjectStoreItem>();
			downLightItem->setName("$variable$downLight$");
			downLightItem->setValueFromBool(downLight);
			assembly.addItem(downLightItem);

			auto sideLightItem = std::make_shared<oso::ObjectStoreItem>();
			sideLightItem->setName("$variable$sideLight$");
			sideLightItem->setValueFromBool(sideLight);
			assembly.addItem(sideLightItem);

			auto exposureTimeItem = std::make_shared<oso::ObjectStoreItem>();
			exposureTimeItem->setName("$variable$exposureTime$");
			exposureTimeItem->setValueFromInt(static_cast<int>(exposureTime));
			assembly.addItem(exposureTimeItem);

			auto gainItem = std::make_shared<oso::ObjectStoreItem>();
			gainItem->setName("$variable$gain$");
			gainItem->setValueFromInt(static_cast<int>(gain));
			assembly.addItem(gainItem);

			auto rootPathItem = std::make_shared<oso::ObjectStoreItem>();
			rootPathItem->setName("$variable$rootPath$");
			rootPathItem->setValueFromString(rootPath);
			assembly.addItem(rootPathItem);

			auto nameItem = std::make_shared<oso::ObjectStoreItem>();
			nameItem->setName("$variable$name$");
			nameItem->setValueFromString(name);
			assembly.addItem(nameItem);

			auto idItem = std::make_shared<oso::ObjectStoreItem>();
			idItem->setName("$variable$id$");
			idItem->setValueFromLong(id);
			assembly.addItem(idItem);

			return assembly;
		}

		bool AiModelConfig::operator==(const AiModelConfig& account) const
		{
			return modelType == account.modelType &&
				date == account.date &&
				upLight == account.upLight &&
				downLight == account.downLight &&
				sideLight == account.sideLight &&
				exposureTime == account.exposureTime &&
				gain == account.gain &&
				rootPath == account.rootPath &&
				name == account.name &&
				id == account.id;
		}

		bool AiModelConfig::operator!=(const AiModelConfig& account) const
		{
			return !(*this == account);
		}
	}
}