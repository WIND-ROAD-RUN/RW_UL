#pragma once

#include"oso_core.h"

#include<string>
#include<filesystem>

namespace rw
{
	namespace oso
	{
		class IStorageStrategy;

		//请使用Xml和Json作为存储格式其他的还未完全实现
		enum class StorageType
		{
			Sqlite,
			Text,
			Binary,
			Json,
			Xml,
			Csv,
			Excel,
			Odbc,
			Oracle,
			Postgres,
			Mysql,
			SqlServer,
			Access,
			Parquet,
			Avro,
			Arrow,
			Hdf5,
			Kafka,
			Redis,
			S3
		};

		enum FileReadResult
		{
			Success,
			BackupUsed,
			NotFound,
			Failed
		};

		class StorageContext
		{
		public:
			explicit StorageContext(StorageType type);
		public:
			bool saveSafe(const ObjectStoreAssembly& assembly, const std::filesystem::path& fileName) const;
			std::shared_ptr<ObjectStoreAssembly> loadSafe(const std::filesystem::path& fileName, FileReadResult & readResult);
			std::shared_ptr<ObjectStoreAssembly> loadSafe(const std::filesystem::path& fileName);
		public:
			bool ensureFileExistsSafe(const std::filesystem::path& fileName, const ObjectStoreAssembly& assembly) const;
			bool ensureFileExists(const std::filesystem::path& fileName, const ObjectStoreAssembly& assembly) const;
		public:
			bool save(const ObjectStoreAssembly& assembly, const std::filesystem::path& fileName) const;
			bool save(const ObjectStoreAssembly& assembly, const std::string& fileName) const;

			[[nodiscard]] std::shared_ptr<ObjectStoreAssembly> load(const std::filesystem::path& fileName) const;
			[[nodiscard]] std::shared_ptr<ObjectStoreAssembly> load(const std::string& fileName) const;

			[[nodiscard]] std::string getFormatString(const ObjectStoreAssembly& assembly) const;
		private:
			std::shared_ptr<IStorageStrategy> _strategy;
		};
	} // namespace oso
} // namespace rw