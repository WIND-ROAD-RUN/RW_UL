#include <QPainter>
#include <QtWidgets/QApplication>
#include"Demo.h"
#include "LicenseValidation.h"


int main(int argc, char* argv[])
{
	QApplication a(argc, argv);


	Demo l;
	l.ini();
	l.show();



	return a.exec();
}
