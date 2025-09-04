#pragma once
#include <memory>
#include <filesystem>
#include <string>

#include "oso_core.h"
#include "oso_StorageContext.hpp"

using Type = rw::oso::ObjectDataItemStoreType;

namespace zzw::XmlMerge
{
	struct XmlMergeTool
	{
	public:
		rw::oso::ObjectStoreAssembly Merge(const std::filesystem::path& defaultXmlPath,
			const std::filesystem::path& existingXmlPath, bool& isSuccess);
	private:
		void Merge(rw::oso::ObjectStoreAssembly& newAssembly,
		           const rw::oso::ObjectStoreAssembly& oldAssembly);

	};
} // namespace zzw::XmlMerge