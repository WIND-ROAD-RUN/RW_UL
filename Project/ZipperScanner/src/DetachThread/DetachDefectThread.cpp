#include "DetachDefectThread.h"

DetachDefectThreadZipper::DetachDefectThreadZipper(QObject* parent)
{

}

DetachDefectThreadZipper::~DetachDefectThreadZipper()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void DetachDefectThreadZipper::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void DetachDefectThreadZipper::stopThread()
{
	running = false; // 停止线程
}

void DetachDefectThreadZipper::processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<DefectValueInfo, DefectValueInfo>>& queue)
{
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	DefectValueInfo preTime;
	try
	{
		//emit findIsBad(1);
	}
	catch (const std::runtime_error&)
	{
		return;
	}
}

void DetachDefectThreadZipper::processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<DefectValueInfo, DefectValueInfo>>& queue)
{
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	DefectValueInfo preTime;
	try
	{

		//emit findIsBad(2);
	}
	catch (const std::runtime_error&)
	{
		return;
	}
}

void DetachDefectThreadZipper::run()
{
	auto& priorityQueue1 = GlobalStructDataZipper::getInstance().priorityQueue1;
	auto& priorityQueue2 = GlobalStructDataZipper::getInstance().priorityQueue2;

	while (running) {
		QThread::msleep(10);
		processQueue1(priorityQueue1);
		processQueue2(priorityQueue2);
	}
}
