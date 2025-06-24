#pragma once

#include <QString>
#include "ZipperScanner.h"

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

using Time = std::chrono::system_clock::time_point;

using DefectValueInfo = Time;

struct WarningId
{
	static constexpr int cairPressureAlarm = 0;
	static constexpr int ccameraDisconnectAlarm1 = 1;
	static constexpr int ccameraDisconnectAlarm2 = 2;
	static constexpr int cworkTrigger1 = 3;
	static constexpr int cworkTrigger2 = 4;
	static constexpr int csportControlAlarm = 5;
	static constexpr int clongTermIdleOperationAlarm = 6;
	static constexpr int cwork1AndWork2EmptyAlarm = 7;
};

struct ControlLines
{
public:
	static size_t qidonganniuIn;
	static size_t lalianlawanIn;
	static size_t jitingIn;
public:
	static size_t bujindianjimaichongOut;
	static size_t chongkongOUT;
	static size_t tuojiOut;
};

struct ClassId
{
	static constexpr int Queya = 0;
	static constexpr int Tangshang = 1;
	static constexpr int Zangwu = 2;
};

inline struct GlobalPath
{
public:
	QString projectHome = R"(D:\zfkjData\ZipperScanner\)";
public:
	QString configRootPath = projectHome + R"(config\)";
	QString modelRootPath = projectHome + R"(model\)";
	QString generalConfigPath = configRootPath + R"(generalConfig.xml)";
	QString scoreConfigPath = configRootPath + R"(scoreConfig.xml)";
	QString setConfigPath = configRootPath + R"(setConfig.xml)";
	QString dlgExposureTimeSetFilePath = configRootPath + R"(dlgExposureTimeSet.xml)";
	QString modelPath = modelRootPath + R"(lalian.engine)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";

}globalPath;
