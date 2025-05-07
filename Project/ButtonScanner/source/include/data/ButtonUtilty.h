#pragma once
#include <QImage>
#include"opencv2/opencv.hpp"
#include<QString>

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

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
	static const int pokong = 8;
	static const int poyan = 9;
	static const int xiaoqikong = 10;
	static const int mofa = 11;
	static const int xiaopobian = 12;
	static const int baibian = 13;

};

inline struct GlobalPath
{
public:
	QString projectHome = R"(D:\zfkjData\ButtonScanner\)";
public:
	QString modelStorageManagerRootPath = projectHome + R"(ModelStorage\)";
	QString modelStorageManagerTempPath = modelStorageManagerRootPath + R"(Temp\)";
public:
	QString configRootPath = projectHome + R"(config\)";
public:
	QString modelRootPath = projectHome + R"(model\)";
	QString engineObb = R"(ObbModel.engine)";
public:
	QString yoloV5RootPath = R"(D:\y\yolov5-master\)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";
}globalPath;
