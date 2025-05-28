#include"demo.h"
#include <QtWidgets/QApplication>
#include"demoConfig.hpp"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	cdm::testClass test;
	rw::oso::ObjectStoreAssembly testAssembly(test);
	cdm::testClass test1(testAssembly);
	auto d = test == test1;


	demo w;
	w.show();

	return a.exec();
}