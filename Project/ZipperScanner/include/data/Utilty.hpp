#pragma once

#include <QString>
#include "ZipperScanner.h"

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

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

struct ControlLine
{
	int axis;
	int ioNum;
	ControlLine(int a, int i)
	{
		axis = a;
		ioNum = i;
	}
};

struct ControlLines
{
public:
	const static ControlLine blowLine1;
	const static ControlLine blowLine2;
public:
	static constexpr int stopIn = 2;
	static constexpr int startIn = 1;
	static constexpr int airWarnIn = 7;
	static constexpr int shutdownComputerIn = 8;
	static constexpr int camer1In = 6;
	static constexpr int camer2In = 5;
public:
	static constexpr int motoPowerOut = 1;
	static constexpr int beltAsis = 0;
	static constexpr int warnGreenOut = 7;
	static constexpr int warnRedOut = 8;
	static constexpr int upLightOut = 9;
	static constexpr int sideLightOut = 0;
	static constexpr int downLightOut = 10;
	static constexpr int strobeLightOut = 11;
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
	QString generalConfigPath = configRootPath + R"(generalConfig.xml)";
	QString scoreConfigPath = configRootPath + R"(scoreConfig.xml)";
	QString setConfigPath = configRootPath + R"(setConfig.xml)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";

}globalPath;
