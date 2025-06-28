#include "stdafx.h"
#include "ButtonScanner.h"
#include <QtWidgets/QApplication>

bool isSingleInstance(const QString& instanceName) {
	static QSharedMemory sharedMemory(instanceName);
	if (!sharedMemory.create(1)) {
		return false; // 已有实例运行
	}
	return true; // 当前实例是唯一的
}

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	a.setWindowIcon(QIcon(":/ButtonScanner/image/icon.png"));

	// 检查单实例
	const QString instanceName = "ButtonScannerInstance";
	if (!isSingleInstance(instanceName)) {
		QMessageBox::critical(nullptr, "错误", "程序已在运行中，无法启动多个实例。");
		return 1; // 退出程序
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