#include "DetachDefectThread.h"

#include <qtconcurrentrun.h>

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

void DetachDefectThreadSmartCroppingOfBags::processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<double, double>>& queue, double nowLocation)
{
	try
	{
		auto & globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		auto prePulse = queue->peek();
		double tempPulse = 0;

		auto duration =std::abs(nowLocation-prePulse);
		auto standard = setConfig.daokoudaoxiangjiluli1 / setConfig.maichongxishu1;
		if (duration> standard)
		{
			auto& motion = globalStruct.zMotion;
			queue->top();
			motion.setIOOut(ControlLines::yadaiOut, true);

			while (true)
			{
				QThread::msleep(1);
				if (motion.getIOOut(ControlLines::qiedaoIn) == false)
				{
					motion.setIOOut(ControlLines::yadaiOut, false);
					break;
				}
			}

			QThread* threaBaojin = QThread::create([&setConfig, &motion]() {
				QThread::msleep(setConfig.baojingyanshi1); // 设置适当的延迟时间
				motion.setIOOut(ControlLines::baojinghongdengOUT, true);
				QThread::msleep(setConfig.baojingshijian1); // 设置适当的延迟时间
				motion.setIOOut(ControlLines::baojinghongdengOUT, false);
				});
			QObject::connect(threaBaojin, &QThread::finished, threaBaojin, &QThread::deleteLater);


			QThread* threadChuiqi = QThread::create([&setConfig, &motion]() {
				QThread::msleep(setConfig.chuiqiyanshi1); // 设置适当的延迟时间
				motion.setIOOut(ControlLines::chuiqiOut, true);
				QThread::msleep(setConfig.chuiqishijian1); // 设置适当的延迟时间
				motion.setIOOut(ControlLines::chuiqiOut, false);
				});
			QObject::connect(threadChuiqi, &QThread::finished, threadChuiqi, &QThread::deleteLater);


			QThread* threadTifei = QThread::create([&setConfig,&motion]() {
				QThread::msleep(setConfig.tifeiyanshi1); // 设置适当的延迟时间
				motion.setIOOut(ControlLines::tifeiOut, true);
				QThread::msleep(setConfig.tifeishijian1); // 设置适当的延迟时间
				motion.setIOOut(ControlLines::tifeiOut, false);
				});
			QObject::connect(threadTifei, &QThread::finished, threadTifei, &QThread::deleteLater);

			threadTifei->start();
			threadChuiqi->start();
			threaBaojin->start();
		}

	}
	catch (const std::runtime_error&)
	{
		return;
	}
}

void DetachDefectThreadSmartCroppingOfBags::run()
{
	while (running) {
		QThread::msleep(4);
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& deque1 = globalStruct.priorityQueue1;
		double location{0};
		auto isGet=globalStruct.camera1->getEncoderNumber(location);
		if (!isGet)
		{
			continue;
		}
		processQueue1(deque1, location);
	}
}
