#pragma once
#include <QImage>
#include"opencv2/opencv.hpp"
#include<QString>

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

struct WarningId
{
	static constexpr int cairPressureAlarm = 0;
	static constexpr int ccameraDisconnectAlarm1 = 1;
	static constexpr int ccameraDisconnectAlarm2 = 2;
	static constexpr int ccameraDisconnectAlarm3 = 3;
	static constexpr int ccameraDisconnectAlarm4 = 4;
	static constexpr int clongTermIdleOperationAlarm = 5;
	static constexpr int cwork1AndWork2EmptyAlarm = 6;
	static constexpr int cwork2AndWork4EmptyAlarm = 7;
	static constexpr int csportControlAlarm = 8;
};

struct ControlLine
{
	int axis;
	int ioNum;
	ControlLine(int a,int i)
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
	const static ControlLine blowLine3;
	const static ControlLine blowLine4;
public:
	static constexpr int stopIn = 2;
	static constexpr int startIn = 1;
	static constexpr int airWarnIN = 7;
	static constexpr int shutdownComputerIn = 8;
	static constexpr int camer1In = 6;
	static constexpr int camer2In = 5;
	static constexpr int camer3In = 4;
	static constexpr int camer4In = 3;
public:
	static constexpr int warnOut = 8;
	static constexpr int motoPowerOut = 1;
	static constexpr int beltAsis = 0;
	static constexpr int warnGreenOut = 7;
	static constexpr int warnRedOut = 8;
	static constexpr int warnUpLightOut = 9;
	static constexpr int warnSideLightOut = 0;
	static constexpr int warnDownLightOut = 10 ;
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
	static constexpr int xiaoqikong = 10;
	static constexpr int mofa = 11;
	static constexpr int xiaopobian = 12;
	static constexpr int baibian = 13;
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
