#include <QtWidgets/QApplication>
#include"rqw_DlgVersion.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	DlgVersion w;
	w.loadVersionPath(R"(D:\zfkjData\SmartCroppingOfBags\Version\Version.txt)");
	w.show();

	return a.exec();
}