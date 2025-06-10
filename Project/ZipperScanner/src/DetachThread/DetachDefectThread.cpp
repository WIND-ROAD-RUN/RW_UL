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

void DetachDefectThreadZipper::processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time>>& queue)
{
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	Time preTime;
	try
	{
		preTime = queue->peek();
		auto now = std::chrono::system_clock::now();

		// 计算时间差
		auto duration = now - preTime;
		auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

		if (milliseconds >= setConfig.yanChiTiFeiShiJian1)
		{
			queue->top(); // 取出队首元素
			emit findIsBad(1);
		}
	}
	catch (const std::runtime_error&)
	{
		return;
	}
}

void DetachDefectThreadZipper::processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time>>& queue)
{
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	Time preTime;
	try
	{
		preTime = queue->peek();
		auto now = std::chrono::system_clock::now();

		// 计算时间差
		auto duration = now - preTime;
		auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

		if (milliseconds >= setConfig.yanChiTiFeiShiJian2)
		{
			queue->top(); // 取出队首元素
			emit findIsBad(2);
		}
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
		QThread::msleep(100);
		processQueue1(priorityQueue1);
		processQueue2(priorityQueue2);
	}
}
