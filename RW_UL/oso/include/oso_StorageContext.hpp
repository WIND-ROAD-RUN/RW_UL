#pragma once

#include"oso_core.h"

#include<string>
#include<filesystem>

namespace rw
{
    namespace oso
    {
        class IStorageStrategy;

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

        class StorageContext
        {
        public:
            explicit StorageContext(StorageType type);
        public:
            void save(const ObjectStoreAssembly& assembly, const std::filesystem::path& fileName) const;
            void save(const ObjectStoreAssembly& assembly, const std::string & fileName) const;

            [[nodiscard]] std::shared_ptr<ObjectStoreAssembly> load(const std::filesystem::path& fileName) const;
            [[nodiscard]] std::shared_ptr<ObjectStoreAssembly> load(const std::string& fileName) const;

            [[nodiscard]] std::string getFormatString(const ObjectStoreAssembly& assembly) const;
        private:
            std::shared_ptr<IStorageStrategy> _strategy;
        };


    } // namespace oso
} // namespace rw