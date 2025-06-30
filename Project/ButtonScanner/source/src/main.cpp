#include "stdafx.h"
#include "ButtonScanner.h"
#include <QtWidgets/QApplication>

#include"RunEnvCheck.hpp"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	a.setWindowIcon(QIcon(":/ButtonScanner/image/icon.png"));

	if (RunEnvCheck::envCheck() ==EnvCheckResult::EnvError)
	{
		return 1;
	}

	ButtonScanner w;
	w.setFixedSize(w.size());

#ifdef NDEBUG
	w.showFullScreen();
#else
	w.show();
#endif

	return a.exec();
}