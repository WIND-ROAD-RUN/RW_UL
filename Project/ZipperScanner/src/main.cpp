#include "ZipperScanner.h"
#include <QtWidgets/QApplication>
int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	ZipperScanner w;
	w.show();

	return a.exec();
}