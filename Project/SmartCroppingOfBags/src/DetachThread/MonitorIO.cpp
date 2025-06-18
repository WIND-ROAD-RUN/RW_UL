#include "MonitorIO.h"

MonitorIOSmartCroppingOfBags::MonitorIOSmartCroppingOfBags(QObject* parent)
{

}

MonitorIOSmartCroppingOfBags::~MonitorIOSmartCroppingOfBags()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void MonitorIOSmartCroppingOfBags::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void MonitorIOSmartCroppingOfBags::stopThread()
{
	running = false; // 停止线程
}

void MonitorIOSmartCroppingOfBags::run()
{
}
