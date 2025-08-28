#include <iostream>
#include <QPainter>
#include <random>
#include <QtWidgets/QApplication>
#include"DlgCloseForm.h"
#include "rqw_ImageSaveEngineV1.hpp"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	rw::rqw::ImageSaveEngineV1 engine;
	engine.startEngine();

	for (int i=0;i<100000;i++)
	{
		QImage image(640, 640, QImage::Format_RGB32);
		image.fill(Qt::yellow);
		rw::rqw::ImageSaveInfoV1 info(std::move(image));
		info.saveDirectoryPath = R"(C:\Users\rw\Desktop\OK\testDir)";
		engine.pushImage(info);
	}

	engine.stopEngine();

	return a.exec();
}