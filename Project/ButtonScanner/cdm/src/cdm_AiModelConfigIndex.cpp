#include"cdm_AiModelConfigIndex.h"

#include"oso_core.h"

namespace rw
{
	namespace cdm
	{
		ConfigIndexItem::ConfigIndexItem(const ConfigIndexItem& item)
		{
			root_path = item.root_path;
			model_name = item.model_name;
			model_type = item.model_type;
			id = item.id;
		}

		ConfigIndexItem::ConfigIndexItem(const rw::oso::ObjectStoreAssembly& assembly)
		{
			auto isAccountAssembly = assembly.getName();
			if (isAccountAssembly != "$class$ConfigIndexItem$") {
				throw std::runtime_error("Assembly is not $class$ConfigIndexItem$");
			}
			auto rootPathItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$root_path$"));
			if (!rootPathItem) {
				throw std::runtime_error("$variable$root_path is not found");
			}
			root_path = rootPathItem->getValueAsString();
			auto modelNameItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$model_name$"));
			if (!modelNameItem) {
				throw std::runtime_error("$variable$model_name is not found");
			}
			model_name = modelNameItem->getValueAsString();
			auto modelTypeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$model_type$"));
			if (!modelTypeItem) {
				throw std::runtime_error("$variable$model_type is not found");
			}
			model_type = IntToModelType(modelTypeItem->getValueAsInt());
			auto idTypeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$id$"));
			if (!idTypeItem) {
				throw std::runtime_error("$variable$id is not found");
			}
			id = idTypeItem->getValueAsLong();
		}

		ConfigIndexItem::operator oso::ObjectStoreAssembly() const
		{
			auto assembly = rw::oso::ObjectStoreAssembly();
			assembly.setName("$class$ConfigIndexItem$");
			auto rootPathItem = rw::oso::ObjectStoreItem();
			rootPathItem.setName("$variable$root_path$");
			rootPathItem.setValueFromString(root_path);
			assembly.addItem(std::make_shared<rw::oso::ObjectStoreItem>(rootPathItem));
			auto modelNameItem = rw::oso::ObjectStoreItem();
			modelNameItem.setName("$variable$model_name$");
			modelNameItem.setValueFromString(model_name);
			assembly.addItem(std::make_shared<rw::oso::ObjectStoreItem>(modelNameItem));
			auto modelTypeItem = rw::oso::ObjectStoreItem();
			modelTypeItem.setName("$variable$model_type$");
			modelTypeItem.setValueFromInt(ModelTypeToInt(model_type));
			assembly.addItem(std::make_shared<rw::oso::ObjectStoreItem>(modelTypeItem));
			auto idTypeItem = rw::oso::ObjectStoreItem();
			idTypeItem.setName("$variable$id$");
			idTypeItem.setValueFromLong(id);
			assembly.addItem(std::make_shared<rw::oso::ObjectStoreItem>(idTypeItem));
			return assembly;
		}

		bool ConfigIndexItem::operator==(const ConfigIndexItem& item) const
		{
			return root_path == item.root_path &&
				model_name == item.model_name &&
				model_type == item.model_type &&
				id == item.id;
		}

		bool ConfigIndexItem::operator!=(const ConfigIndexItem& item) const
		{
			return !(*this == item);
		}

		AiModelConfigIndex::operator oso::ObjectStoreAssembly() const
		{
			auto assembly = rw::oso::ObjectStoreAssembly();
			assembly.setName("$class$AiModelConfigIndex$");

			for (const auto& item : modelIndexs)
			{
				auto childAssembly = item.operator rw::oso::ObjectStoreAssembly();
				assembly.addItem(childAssembly);
			}

			return assembly;
		}

		bool AiModelConfigIndex::operator==(const AiModelConfigIndex& account) const
		{
			if (modelIndexs.size() != account.modelIndexs.size()) {
				return false;
			}
			for (size_t i = 0; i < modelIndexs.size(); ++i) {
				if (modelIndexs[i] != account.modelIndexs[i]) {
					return false;
				}
			}
			return true;
		}

		bool AiModelConfigIndex::operator!=(const AiModelConfigIndex& account) const
		{
			return !(*this == account);
		}

		AiModelConfigIndex::AiModelConfigIndex(const rw::oso::ObjectStoreAssembly& assembly)
		{
			auto isAccountAssembly = assembly.getName();
			if (isAccountAssembly != "$class$AiModelConfigIndex$") {
				throw std::runtime_error("Assembly is not $class$AiModelConfigIndex$");
			}
			auto items = assembly.getItems();
			for (const auto& item : items) {
				if (item->getStoreType() == "assembly") {
					auto childAssembly = std::dynamic_pointer_cast<rw::oso::ObjectStoreAssembly>(item);
					if (childAssembly) {
						modelIndexs.push_back(ConfigIndexItem(*childAssembly));
					}
				}
			}
		}

		AiModelConfigIndex::AiModelConfigIndex(const AiModelConfigIndex& buttonScannerMainWindow)
		{
			modelIndexs = buttonScannerMainWindow.modelIndexs;
		}

		AiModelConfigIndex& AiModelConfigIndex::operator=(const AiModelConfigIndex& buttonScannerMainWindow)
		{
			if (this != &buttonScannerMainWindow) {
				modelIndexs = buttonScannerMainWindow.modelIndexs;
			}
			return *this;
		}

		void AiModelConfigIndex::pushConfig(const ConfigIndexItem& item)
		{
			for (const auto& index : modelIndexs)
			{
				if (index.id == item.id)
				{
					return;
				}
			}
			modelIndexs.push_back(item);
		}

		void AiModelConfigIndex::deleteConfig(const ConfigIndexItem& item)
		{
			for (size_t i = 0; i < modelIndexs.size(); ++i)
			{
				if (modelIndexs[i].id == item.id)
				{
					modelIndexs.erase(modelIndexs.begin() + i);
					return;
				}
			}
		}
	}
}