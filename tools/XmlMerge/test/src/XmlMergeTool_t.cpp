#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>

#include "XmlMergeTool_t.hpp"
#include "oso_core.h"
#include "oso_StorageContext.hpp"


using namespace rw::oso;
using zzw::XmlMerge::XmlMergeTool;

static const std::filesystem::path kNewVersionXmlPath = std::filesystem::path(
	R"(D:\hagkData\SmartCroppingOfBags\config\generalConfig.xml)");

static const std::filesystem::path kOldVersionXmlPath = std::filesystem::path(
	R"(C:\Users\zfkj4090\Desktop\generalConfig.xml)");

static const std::filesystem::path kMergedOutputPath = std::filesystem::path(
	R"(C:\Users\zfkj4090\Desktop\xmltest\generalConfig.xml)");


TEST(XmlMergeRealFile, MergeTwoRealXmlFiles)
{
	auto newPath = kNewVersionXmlPath;
	auto oldPath = kOldVersionXmlPath;

	auto equalPath = kMergedOutputPath;

	StorageContext newStorage(StorageType::Xml);


}