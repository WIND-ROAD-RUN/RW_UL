#pragma once

#include <QString>
#include "ZipperScanner.h"

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

struct WarningId
{
	static constexpr int ccameraDisconnectAlarm1 = 1;
	static constexpr int ccameraDisconnectAlarm2 = 2;
};

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
