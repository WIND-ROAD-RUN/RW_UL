#pragma once
#include <QImage>
#include"opencv2/opencv.hpp"
#include<QString>

QImage cvMatToQImage(const cv::Mat& mat);

QPixmap cvMatToQPixmap(const cv::Mat& mat);

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
	QString nameFileName = R"(index.names)";
	QString engineFileName = R"(model.engine)";
	QString onnxFileName = R"(modelOnnx.onnx)";
	QString onnxFileNameOO = R"(customOO.onnx)";
	QString onnxFileNameSO = R"(customSO.onnx)";
public:
	QString yoloV5RootPath = R"(D:\y\yolov5-master\)";
public:
	QString imageSaveRootPath = projectHome + R"(SavedImages\)";
}globalPath;
