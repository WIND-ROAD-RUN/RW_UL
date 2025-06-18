#include "DetachUtiltyThread.h"

#include "GlobalStruct.hpp"
#include "scc_motion.h"

DetachUtiltyThreadSmartCroppingOfBags::DetachUtiltyThreadSmartCroppingOfBags(QObject* parent)
	: QThread(parent), running(false) {

}

DetachUtiltyThreadSmartCroppingOfBags::~DetachUtiltyThreadSmartCroppingOfBags()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void DetachUtiltyThreadSmartCroppingOfBags::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void DetachUtiltyThreadSmartCroppingOfBags::stopThread()
{
	running = false; // 停止线程
}

void DetachUtiltyThreadSmartCroppingOfBags::run()
{
	static size_t s = 0;
	while (running) {
		QThread::sleep(1);
		processTrigger(s);
		++s;
		if (s == 300)
		{
			s = 0;
		}
	}
}

void DetachUtiltyThreadSmartCroppingOfBags::processTrigger(size_t s)
{
	
}



