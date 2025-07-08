#include <QPainter>
#include <QtWidgets/QApplication>
#include"Demo.h"
#include "LicenseValidation.h"


int main(int argc, char* argv[])
{
	QApplication a(argc, argv);


	Demo l;
	
	l.show();
	l.ini();


	return a.exec();
}
