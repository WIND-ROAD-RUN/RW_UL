#include "SmartCroppingOfBags.h"
#include "DlgProductSet.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	SmartCroppingOfBags w;
	//w.showFullScreen();
	w.setFixedSize(w.size());
	w.show();

	return a.exec();
}