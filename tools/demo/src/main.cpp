
#include <QtWidgets/QApplication>
#include"PictureViewerThumbnails.h"


int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	PictureViewerThumbnails w;
	w.setRootPath(R"(C:\Users\rw\Desktop\temp2)");
	w.show();

	return a.exec();
}