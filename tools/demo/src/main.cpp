#include <QPainter>
#include <QtWidgets/QApplication>
#include"rqw_DlgVersion.h"

#include"PictureViewerThumbnails.h"
int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	PictureViewerThumbnails p;
	p.setRootPath(R"(C:\Users\rw\Desktop\temp)");
	p.show();
	

	return a.exec();
}
