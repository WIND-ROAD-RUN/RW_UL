#pragma once

#include <QString>
#include "ZipperScanner.h"

inline struct GlobalPath
{
public:
	QString projectHome = R"(D:\zfkjData\ZipperScanner\)";
public:
	QString configRootPath = projectHome + R"(config\)";
	QString generalConfigPath = configRootPath + R"(generalConfig.xml)";
	QString scoreConfigPath = configRootPath + R"(scoreConfig.xml)";
	QString setConfigPath = configRootPath + R"(setConfig.xml)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";

}globalPath;
