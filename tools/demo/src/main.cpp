#include <iostream>
#include <QPainter>
#include <random>
#include <QtWidgets/QApplication>
#include"DlgCloseForm.h"
#include "rqw_ImageSaveEngineV1.hpp"
#include"PicturesPainterVersionDunDai.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	/*rw::rqw::ImageSaveEngineV1 engine;
	engine.startEngine();

	for (int i=0;i<100000;i++)
	{
		QImage image(640, 640, QImage::Format_RGB32);
		image.fill(Qt::yellow);
		rw::rqw::ImageSaveInfoV1 info(std::move(image));
		info.saveDirectoryPath = R"(C:\Users\rw\Desktop\OK\testDir)";
		engine.pushImage(info);
	}

	engine.stopEngine();*/

	PicturesPainterVersionDunDai painter;
	QImage image(R"(C:\Users\rw\Desktop\temp\2025-06-17_11-10-55-662.jpg)");
	//QImage image(R"(C:\Users\rw\Desktop\temp\total.png)");
	std::vector<rw::rqw::RectangeConfig> cfgs;
	rw::rqw::RectangeConfig config;
	config.classid = 0;
	config.color = QColor(255, 0, 0);
	config.name = "name_0";
	cfgs.push_back(config);
	painter.setRectangleConfigs(cfgs);
	painter.setImage(image);
	painter.show();

	return a.exec();
}