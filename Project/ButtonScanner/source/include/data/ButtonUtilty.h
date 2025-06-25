#pragma once
#include <QImage>
#include"opencv2/opencv.hpp"
#include<QString>

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

struct VersionInfo
{
	static QString Version;
};

struct WarningId
{
	static constexpr int cairPressureAlarm = 0;
	static constexpr int ccameraDisconnectAlarm1 = 1;
	static constexpr int ccameraDisconnectAlarm2 = 2;
	static constexpr int ccameraDisconnectAlarm3 = 3;
	static constexpr int ccameraDisconnectAlarm4 = 4;
	static constexpr int cworkTrigger1 = 5;
	static constexpr int cworkTrigger2 = 6;
	static constexpr int cworkTrigger3 = 7;
	static constexpr int cworkTrigger4 = 8;
	static constexpr int csportControlAlarm = 9;
	static constexpr int clongTermIdleOperationAlarm = 10;
	static constexpr int cwork1AndWork2EmptyAlarm = 11;
	static constexpr int cwork2AndWork4EmptyAlarm = 12;
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
	static ControlLine blowLine1;
	static ControlLine blowLine2;
	static ControlLine blowLine3;
	static ControlLine blowLine4;
public:
	static int stopIn;
	static int startIn;
	static int airWarnIn;
	static int shutdownComputerIn;
	static int camer1In;
	static int camer2In;
	static int camer3In;
	static int camer4In;
public:
	static int motoPowerOut;
	static int beltAsis;
	static int warnGreenOut;
	static int warnRedOut;
	static int upLightOut;
	static int sideLightOut;
	static int downLightOut;
	static int strobeLightOut;
};

struct ClassId
{
	static constexpr int Body = 0;
	static constexpr int Hole = 1;
	static constexpr int pobian = 2;
	static constexpr int qikong = 3;
	static constexpr int duyan = 4;
	static constexpr int moshi = 5;
	static constexpr int liaotou = 6;
	static constexpr int zangwu = 7;
	static constexpr int liehen = 8;
	static constexpr int poyan = 9;
	static constexpr int smallPore = 11;
};

struct ClassIdPositive
{
	static const int Good = 0;
	static const int Bad = 1;
};

inline struct GlobalPath
{
public:
	QString projectHome = R"(D:\zfkjData\ButtonScanner\)";
public:
	QString modelStorageManagerRootPath = projectHome + R"(ModelStorage\)";
	QString modelStorageManagerTempPath = modelStorageManagerRootPath + R"(Temp\)";
	QString trainAIRootPath = projectHome + R"(Train\)";
	QString trainAIObbRootPath = projectHome + R"(Train\Obb\)";
	QString trainAISegRootPath = projectHome + R"(Train\Seg\)";
public:
	QString configRootPath = projectHome + R"(config\)";
public:
	QString modelRootPath = projectHome + R"(model\)";
	QString engineObb = R"(ObbModel.engine)";
	QString engineSeg = R"(SegModel.engine)";
	QString onnxRuntime1 = R"(customOO1.engine)";
	QString onnxRuntime2 = R"(customOO2.engine)";
	QString onnxRuntime3 = R"(customOO3.engine)";
	QString onnxRuntime4 = R"(customOO4.engine)";
public:
	QString yoloV5RootPath = R"(D:\y\yolov5-master\)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";
}globalPath;
