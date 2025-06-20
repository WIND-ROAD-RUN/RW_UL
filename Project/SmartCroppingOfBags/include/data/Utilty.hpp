#pragma once

#include <iostream>
#include <QString>
#include "smartCroppingOfBags.h"

using Time = std::chrono::system_clock::time_point;

inline void printTimeWithMilliseconds(const std::chrono::system_clock::time_point& timePoint) {
	// 转换为 time_t 类型
	std::time_t timeT = std::chrono::system_clock::to_time_t(timePoint);

	// 获取毫秒部分
	auto duration = timePoint.time_since_epoch();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration) % 1000;

	// 格式化输出时间
	std::cout << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S")
		<< '.' << std::setfill('0') << std::setw(3) << milliseconds.count()
		<< std::endl;
}


struct MonitorRunningStateInfo
{
	double averagePulse{0};
	bool isGetAveragePulse{false};

	double currentPulse{0};
	bool isGetCurrentPulse{ false };

	double averagePulseBag{0};
	bool isGetAveragePulseBag{ false };

	double averagePixelBag{0};
	bool isGetAveragePixelBag{ false };

	double lineHeight{0};
	bool isGetLineHeight{ false };
};

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