
#include <QtWidgets/QApplication>
#include"PictureViewerThumbnails.h"


int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	PictureViewerThumbnails w;
	w.setRootPath(R"(C:\Users\zfkj\Desktop\temp\OK\images\123)");
	w.show();

	return a.exec();
}