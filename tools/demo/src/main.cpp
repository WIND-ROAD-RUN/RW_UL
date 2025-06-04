#include"demo.h"
#include <QtWidgets/QApplication>
#include"demoConfig.hpp"
#include"ThumbnailsViewer.hpp"
#include"opencv2/opencv.hpp"
#include"PictureViewerThumbnails.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	/*cdm::testClass test;
	rw::oso::ObjectStoreAssembly testAssembly(test);
	cdm::testClass test1(testAssembly);
	auto d = test == test1;

	ThumbnailsViewer viewer;
	viewer.setRootPath(R"(C:\Users\rw\Desktop\1)");
	viewer.setSize(QSize(500,500));
	viewer.show();*/

	PictureViewerThumbnails viewer;
	viewer.setRootPath(R"(D:\zfkjData\ButtonScanner\ModelStorage\Temp\Image)");
	viewer.setSize(QSize(400, 400));
	viewer.show();

	return a.exec();
}