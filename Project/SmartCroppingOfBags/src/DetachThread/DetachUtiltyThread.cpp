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
		getRunningState(s);
		++s;
		if (s == 300)
		{
			s = 0;
		}
	}
}


void DetachUtiltyThreadSmartCroppingOfBags::getRunningState(size_t s)
{
	if (s % 1 == 0)
	{
		MonitorRunningStateInfo info;
		info.currentPulse = getPulse(info.isGetCurrentPulse);
		info.averagePixelBag = getAveragePixelBag(info.isGetAveragePixelBag);
		info.averagePulse = getAveragePulse(info.isGetAveragePulse);
		info.averagePulseBag = getAveragePulseBag(info.isGetAveragePulseBag);
		info.lineHeight = getLineHeight(info.isGetLineHeight);
		emit updateMonitorRunningStateInfo(info);
	}
}

double DetachUtiltyThreadSmartCroppingOfBags::getPulse(bool& isGet)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	if (!globalStruct.camera1)
	{
		isGet = false;
		return 0;
	}

	if (!globalStruct.camera1->getConnectState())
	{
		isGet = false;
		return 0;
	}

	double pulse{ 0 };
	auto isGetPulse = globalStruct.camera1->getEncoderNumber(pulse);
	if (isGetPulse)
	{
		isGet = true;
		return pulse;
	}
	isGet = false;
	return 0;
}

double DetachUtiltyThreadSmartCroppingOfBags::getAveragePulse(bool& isGet)
{
	isGet = true;
	return pulseAverage;
}

double DetachUtiltyThreadSmartCroppingOfBags::getAveragePulseBag(bool& isGet)
{
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double maichongxishu = setConfig.maichongxishu1;
	isGet = true;
	return maichongxishu * pulseAverage;
}

double DetachUtiltyThreadSmartCroppingOfBags::getAveragePixelBag(bool& isGet)
{
	return 0;
}

double DetachUtiltyThreadSmartCroppingOfBags::getLineHeight(bool& isGet)
{
	return 0;
}

void DetachUtiltyThreadSmartCroppingOfBags::onAppendPulse(double pulse)
{
	lastPulse = pulse; // 更新上次脉冲值
	pulse = pulse - lastPulse; // 计算当前脉冲与上次脉冲的差值
	// 累加所有历史脉冲差值
	pulseSum += pulse;
	++pulseCount;

	pulseAverage = (pulseCount == 0) ? 0.0 : (pulseSum / pulseCount);
}



