#include "PicturesPainter.h"
#include <QtWidgets/QApplication>

#include "rqw_rqwColor.hpp"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	PicturesPainter w;
	QColor color = rw::rqw::RQWColorToQColor(rw::rqw::RQWColor::Blue);
	std::vector<rw::rqw::RectangeConfig> configs;
	rw::rqw::RectangeConfig config1;
	config1.classid = 1;
	config1.color = color;
	config1.name = "Example Class";
	config1.descrption = "This is an example rectangle configuration.";
	configs.push_back(config1);
	w.setRectangleConfigs(configs);
	w.show();

	return a.exec();
}
