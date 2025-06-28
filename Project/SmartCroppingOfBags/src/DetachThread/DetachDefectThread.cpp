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
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;

	double bottomLocation{0};
	try
	{
		bottomLocation = queue->peek();
		auto duration =std::abs(nowLocation-bottomLocation);
		auto standard = setConfig.daokoudaoxiangjiluli1 / setConfig.maichongxishu1;
		if (duration> standard)
		{
			queue->top();
			emitErrorToZMotion();
			globalStruct.locations->clear();
			std::cout << "************************" << std::endl;
		}
		else
		{
			auto& globalThread = GlobalStructThreadSmartCroppingOfBags::getInstance();
			if (globalThread.isQieDao)
			{
				auto topLocationOption=globalStruct.locations->get(bottomLocation);
				if (!topLocationOption.has_value())
				{
					return;
				}

				if (topLocationOption> globalThread.currentQieDaoLocation)
				{
					std::cout << "----------------------" << std::endl;
					queue->top();
					emitErrorToZMotion();
				}

			}
		}

	}
	catch (const std::runtime_error&)
	{
		return;
	}
}

void DetachDefectThreadSmartCroppingOfBags::emitErrorToZMotion()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;

	auto& motion = globalStruct.zMotion;
	auto setResult=motion.setIOOut(ControlLines::yadaiOut, true);

	while (true)
	{
		QThread::msleep(1);
		if (motion.getIOOut(ControlLines::qiedaoIn) == false)
		{
			auto setResult = motion.setIOOut(ControlLines::yadaiOut, false);
			break;
		}
	}

	static std::atomic_bool isFinshBaojing;
	QThread* threaBaojin = QThread::create([&setConfig, &motion]() {
		if (isFinshBaojing)
		{
			QThread::msleep(setConfig.baojingyanshi1); // 设置适当的延迟时间
			auto setResult = motion.setIOOut(ControlLines::baojinghongdengOUT, true);
			QThread::msleep(setConfig.baojingshijian1); // 设置适当的延迟时间
			setResult=motion.setIOOut(ControlLines::baojinghongdengOUT, false);
			isFinshBaojing = true;
		}
		isFinshBaojing = false;
		});
	QObject::connect(threaBaojin, &QThread::finished, threaBaojin, &QThread::deleteLater);

	static std::atomic_bool isFinshChuiqi{ true };
	QThread* threadChuiqi = QThread::create([&setConfig, &motion]() {
		if (isFinshChuiqi)
		{
			QThread::msleep(setConfig.chuiqiyanshi1); // 设置适当的延迟时间
			auto setResult = motion.setIOOut(ControlLines::chuiqiOut, true);
			QThread::msleep(setConfig.chuiqishijian1); // 设置适当的延迟时间
			setResult=motion.setIOOut(ControlLines::chuiqiOut, false);
			isFinshChuiqi = true;
		}
		isFinshChuiqi = false;
		});
	QObject::connect(threadChuiqi, &QThread::finished, threadChuiqi, &QThread::deleteLater);

	static std::atomic_bool isFinshTifei{ true };
	QThread* threadTifei = QThread::create([&setConfig, &motion]() {
		if (isFinshTifei)
		{
			QThread::msleep(setConfig.tifeiyanshi1); // 设置适当的延迟时间
			auto setResult = motion.setIOOut(ControlLines::tifeiOut, true);
			QThread::msleep(setConfig.tifeishijian1); // 设置适当的延迟时间
			setResult=motion.setIOOut(ControlLines::tifeiOut, false);
			isFinshTifei = true;
		}
		isFinshTifei = false;
		});
	QObject::connect(threadTifei, &QThread::finished, threadTifei, &QThread::deleteLater);

	threadTifei->start();
	threadChuiqi->start();
	threaBaojin->start();
}

void DetachDefectThreadSmartCroppingOfBags::run()
{
	while (running) {
		QThread::msleep(4);
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& deque1 = globalStruct.priorityQueue1;
		double location{0};
		if (globalStruct.camera1)
		{
			auto isGet = globalStruct.camera1->getEncoderNumber(location);
			if (!isGet)
			{
				continue;
			}
		}
		
		processQueue1(deque1, location);
	}
}
