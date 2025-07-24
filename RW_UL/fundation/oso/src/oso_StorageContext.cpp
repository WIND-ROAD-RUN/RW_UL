#include"oso_StorageContext.hpp"

#include"oso_StorageStrategy_Xml.hpp"
#include"oso_StorageStrategy_Json.hpp"
#include "oso_StorageStrategy_Sqlite.hpp"
#include"oso_StorageStrategy_Text.hpp"
#include"oso_IStorageStrategy.hpp"

namespace rw
{
	namespace oso
	{
		StorageContext::StorageContext(StorageType type)
		{
			switch (type)
			{
			case StorageType::Xml:
				_strategy = std::make_shared<StorageStrategy_Xml>();
				break;
			case StorageType::Json:
				_strategy = std::make_shared<StorageStrategy_Json>();
				break;
			case StorageType::Text:
				_strategy = std::make_shared<StorageStrategy_Text>();
				break;
			case StorageType::Sqlite:
				_strategy = std::make_shared<StorageStrategy_Sqlite>();
				break;
			default:
				break;
			}
		}

		bool StorageContext::saveSafe(const ObjectStoreAssembly& assembly, const std::filesystem::path& fileName) const
		{
			std::filesystem::path tmpFile = fileName;
			tmpFile += ".tmp";
			if (!_strategy->save(assembly, tmpFile)) {
				return false;
			}

			if (std::filesystem::exists(fileName)) {
				std::filesystem::path bakFile = fileName;
				bakFile += ".bak";
				std::error_code ec;
				std::filesystem::copy_file(fileName, bakFile, std::filesystem::copy_options::overwrite_existing, ec);
			}

			std::error_code ec;
			std::filesystem::rename(tmpFile, fileName, ec);
			if (ec) {
				std::filesystem::remove(tmpFile);
				return false;
			}

			return true;
		}

		std::shared_ptr<ObjectStoreAssembly> StorageContext::loadSafe(const std::filesystem::path& fileName,
			FileReadResult& readResult)
		{
			auto result = _strategy->load(fileName);
			if (result) {
				readResult = FileReadResult::Success;
				return result;
			}

			std::filesystem::path bakFile = fileName;
			bakFile += ".bak";
			if (std::filesystem::exists(bakFile)) {
				result = _strategy->load(bakFile);
				if (result) {
					readResult = FileReadResult::BackupUsed;
					return result;
				}
			}

			if (!std::filesystem::exists(fileName) && !std::filesystem::exists(bakFile)) {
				readResult = FileReadResult::NotFound;
			}
			else {
				readResult = FileReadResult::Failed;
			}
			return nullptr;
		}

		std::shared_ptr<ObjectStoreAssembly> StorageContext::loadSafe(const std::filesystem::path& fileName)
		{
			FileReadResult readResult;
			return loadSafe(fileName, readResult);
		}

		bool StorageContext::ensureFileExists(const std::filesystem::path& fileName, const ObjectStoreAssembly& assembly) const
		{
			if (std::filesystem::exists(fileName)) {
				return true;
			}
			return _strategy->save(assembly, fileName);
		}

		bool StorageContext::save(const ObjectStoreAssembly& assembly, const std::filesystem::path& fileName) const
		{
			return _strategy->save(assembly, fileName);
		}

		bool StorageContext::save(const ObjectStoreAssembly& assembly, const std::string& fileName) const
		{
			return _strategy->save(assembly, fileName);
		}

		std::shared_ptr<ObjectStoreAssembly> StorageContext::load(const std::filesystem::path& fileName) const
		{
			return _strategy->load(fileName);
		}

		std::shared_ptr<ObjectStoreAssembly> StorageContext::load(const std::string& fileName) const
		{
			return _strategy->load(fileName);
		}

		std::string StorageContext::getFormatString(const ObjectStoreAssembly& assembly) const
		{
			return _strategy->getFormatString(assembly);
		}
	} // namespace oso
} // namespace rw