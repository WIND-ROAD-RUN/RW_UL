#include "DetachDefectThread.h"

#include "GlobalStruct.hpp"

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

void DetachDefectThreadSmartCroppingOfBags::processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<double, double>>& queue)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	double prePulse,nowPulse;
	try
	{
		prePulse = queue->peek();
		double tempPulse = 0;
		globalStruct.camera1->getEncoderNumber(tempPulse);
		nowPulse = static_cast<double>(tempPulse);

		// 计算脉冲差
		auto duration = nowPulse - prePulse;

		if (false)
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

void DetachDefectThreadSmartCroppingOfBags::processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<double, double>>& queue)
{

}

void DetachDefectThreadSmartCroppingOfBags::run()
{
	static size_t s = 0;
	while (running) {
		QThread::sleep(1);
		++s;
		if (s == 300)
		{
			s = 0;
		}
	}
}
