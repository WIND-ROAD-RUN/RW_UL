#pragma once
#include <QImage>
#include"opencv2/opencv.hpp"
#include<QString>

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

struct BlowLine
{
	int axis;
	int ioNum;
	BlowLine(int a,int i)
	{
		axis = a;
		ioNum = i;
	}
};

struct BlowLines
{
	const static BlowLine blowLine1;
	const static BlowLine blowLine2;
	const static BlowLine blowLine3;
	const static BlowLine blowLine4;
};

struct ClassId
{
	static const int Body = 0;
	static const int Hole = 1;
	static const int pobian = 2;
	static const int qikong = 3;
	static const int duyan = 4;
	static const int moshi = 5;
	static const int liaotou = 6;
	static const int zangwu = 7;
	static const int liehen = 8;
	static const int poyan = 9;
	static const int xiaoqikong = 10;
	static const int mofa = 11;
	static const int xiaopobian = 12;
	static const int baibian = 13;
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
