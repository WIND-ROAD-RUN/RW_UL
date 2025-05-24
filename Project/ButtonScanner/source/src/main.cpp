#include "stdafx.h"
#include "ButtonScanner.h"
#include <QtWidgets/QApplication>
int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	a.setWindowIcon(QIcon(":/ButtonScanner/image/icon.png"));

	ButtonScanner w;
	//w.showFullScreen();
	w.setFixedSize(w.size());
	w.show();

	return a.exec();
}