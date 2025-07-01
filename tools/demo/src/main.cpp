#include <QPainter>
#include <QtWidgets/QApplication>

#include"PictureViewerThumbnails.h"
#include "FullKeyBoard.h"
int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	FullKeyBoard p;
	p.setWindowFlags(Qt::FramelessWindowHint);
	p.show();
	

	return a.exec();
}
