#include "AutomaticAnnotation.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	AutomaticAnnotation w;
	w.show();

	return a.exec();
}
