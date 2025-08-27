#include <iostream>
#include <QPainter>
#include <random>
#include <QtWidgets/QApplication>
#include"DlgCloseForm.h"
#include "rqw_ImageSaveEngineRefactor.hpp"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	QImage image(640, 640, QImage::Format_RGB32);
	image.fill(Qt::yellow);
	rw::rqw::ImageSaveInfoRefactor info(image);
	info.saveDirectoryPath = R"(C:\Users\rw\Desktop\OK\testDir)";

	rw::rqw::ImageSaveEngineRefactor engine;
	engine.setRootPath(R"(C:\Users\rw\Desktop\OK\test)");
	engine.startEngine();
	engine.pushImage(info);
	engine.stopEngine();

	return a.exec();
}