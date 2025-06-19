#pragma once

#include <QString>
#include "smartCroppingOfBags.h"

using Time = std::chrono::system_clock::time_point;

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
	static size_t qiedaoIn;
public:
	static size_t chuiqiOut;
	static size_t baojinghongdengOUT;
	static size_t yadaiOut;
	static size_t tifeiOut;
};

struct ClassId
{
	static constexpr int Heiba = 0;
	static constexpr int Shudang = 1;
	static constexpr int Huapo = 2;
	static constexpr int Jietou = 3;
	static constexpr int Guasi = 4;
	static constexpr int Podong = 5;
	static constexpr int Zangwu = 6;
	static constexpr int Noshudang = 7;
	static constexpr int Xiaopodong = 8;
	static constexpr int Jiaodai = 9;
	static constexpr int Yinshuaquexian = 10;
	static constexpr int Modian = 11;
	static constexpr int Loumo = 12;
	static constexpr int Xishudang = 13;
	static constexpr int Erweima = 14;
	static constexpr int Damodian = 15;
	static constexpr int Kongdong = 16;
	static constexpr int Sebiao = 17;
};

inline struct GlobalPath
{
public:
	QString projectHome = R"(D:\zfkjData\SmartCroppingOfBags\)";
public:
	QString configRootPath = projectHome + R"(config\)";
	QString modelRootPath = projectHome + R"(model\)";
	QString generalConfigPath = configRootPath + R"(generalConfig.xml)";
	QString scoreConfigPath = configRootPath + R"(scoreConfig.xml)";
	QString setConfigPath = configRootPath + R"(setConfig.xml)";
	QString dlgExposureTimeSetFilePath = configRootPath + R"(dlgExposureTimeSet.xml)";
	QString modelPath = modelRootPath + R"(dundai.engine)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";

}globalPath;