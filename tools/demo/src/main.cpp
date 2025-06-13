
#include <QtWidgets/QApplication>
#include"rqw_CarouselWidget.h"


int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	CarouselWidget w;
	w.appendItem(1);
	w.appendItem(1);
	w.appendItem(2);
	w.appendItem(1);
	w.dequeItem();
	w.show();

	return a.exec();
}