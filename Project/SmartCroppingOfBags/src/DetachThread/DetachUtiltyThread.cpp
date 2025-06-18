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
		getMaiChongXinhao(s);
		++s;
		if (s == 300)
		{
			s = 0;
		}
	}
}

void DetachUtiltyThreadSmartCroppingOfBags::getMaiChongXinhao(size_t s)
{
	if (s%1==0)
	{
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		if (!globalStruct.camera1)
		{
			return;
		}

		if (!globalStruct.camera1->getConnectState())
		{
			return;
		}

		double pulse{0};
		auto isGet=globalStruct.camera1->getEncoderNumber(pulse);
		std::cout << "pulse: " << std::fixed << std::setprecision(2) << pulse << std::endl; // 修改输出格式
		if (isGet)
		{
			emit updateCurrentPulse(pulse);
		}
	}
}




