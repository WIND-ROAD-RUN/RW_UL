#include "ZipperScanner.h"
#include <QtWidgets/QApplication>
int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	ZipperScanner w;
	w.showFullScreen();
	w.setFixedSize(w.size());
	w.show();

	return a.exec();
}