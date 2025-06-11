#include "DetachDefectThread.h"

DetachDefectThreadSmartCroppingOfBags::DetachDefectThreadSmartCroppingOfBags(QObject* parent)
{

}

DetachDefectThreadSmartCroppingOfBags::~DetachDefectThreadSmartCroppingOfBags()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void DetachDefectThreadSmartCroppingOfBags::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void DetachDefectThreadSmartCroppingOfBags::stopThread()
{
	running = false; // 停止线程
}

void DetachDefectThreadSmartCroppingOfBags::processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time>>& queue)
{

}

void DetachDefectThreadSmartCroppingOfBags::processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time>>& queue)
{

}

void DetachDefectThreadSmartCroppingOfBags::run()
{
	QThread::run();

}
