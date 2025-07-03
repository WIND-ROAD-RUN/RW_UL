#include <QPainter>
#include <QtWidgets/QApplication>

#include"PictureViewerThumbnails.h"
#include "FullKeyBoard.h"

#include"LicenseValidation.h"
int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	FullKeyBoard p;
	p.show();
	

	return a.exec();
}
