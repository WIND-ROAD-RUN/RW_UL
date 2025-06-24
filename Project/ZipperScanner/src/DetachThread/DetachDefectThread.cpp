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

void DetachDefectThreadZipper::processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<float, float>>& queue)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& setConfig = globalStruct.setConfig;

	try
	{
		bool isGetLocation = false;
		float nowLocation = globalStruct.zmotion.getAxisLocation(0, isGetLocation);
		auto tifeijuli1 = setConfig.tifeijuli1;
		if (isGetLocation != false && (abs(nowLocation - preLocation) > tifeijuli1))
		{
			//冲孔
			bool isSuccess = globalStruct.zmotion.setIOOut(ControlLines::chongkongOUT,true);
			//emit findIsBad(1);
		}
	}
	catch (const std::runtime_error&)
	{
		return;
	}
}

void DetachDefectThreadZipper::processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<float, float>>& queue)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& setConfig = globalStruct.setConfig;

	try
	{
		bool isGetLocation = false;
		float nowLocation = globalStruct.zmotion.getAxisLocation(0, isGetLocation);
		auto tifeijuli2 = setConfig.tifeijuli2;
		if (isGetLocation != false && (abs(nowLocation - preLocation) > tifeijuli2))
		{
			//冲孔
			bool isSuccess = globalStruct.zmotion.setIOOut(ControlLines::chongkongOUT, true);
			//emit findIsBad(2);
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
		QThread::msleep(10);
		processQueue1(priorityQueue1);
		processQueue2(priorityQueue2);
	}
}
