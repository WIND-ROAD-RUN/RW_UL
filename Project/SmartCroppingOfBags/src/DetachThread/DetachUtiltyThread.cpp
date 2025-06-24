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
	if (s % 1 == 0 &&GlobalStructThreadSmartCroppingOfBags::getInstance()._isUpdateMonitoyInfo)
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

void DetachUtiltyThreadSmartCroppingOfBags::getMainWindowRunningState(size_t s)
{
	auto& info = GlobalStructDataSmartCroppingOfBags::getInstance().statisticalInfo;
	if (s%1==0)
	{
		info.productionYield = info.goodCount / info.produceCount;
		info.averageBagLength = daichangAverageFromPulse;
		updateMainWindowInfo(1);
	}

	if (s%30==0)
	{
		double defectDifference = (info.wasteCount - lastDefectCount) * 2;
		lastDefectCount = info.wasteCount;
		updateMainWindowInfo(30);
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
	daichangAverageFromPulse = maichongxishu * pulseAverage;
	return daichangAverageFromPulse;
}

double DetachUtiltyThreadSmartCroppingOfBags::getAveragePixelBag(bool& isGet)
{
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double daichangxishu = setConfig.daichangxishu1;
	isGet = true;
	daichangAverageFromPixel = daichangxishu * pixelAverage;
	return daichangAverageFromPixel;
}

double DetachUtiltyThreadSmartCroppingOfBags::getLineHeight(bool& isGet)
{
	isGet = true;
	return pixelAverage;
}

void DetachUtiltyThreadSmartCroppingOfBags::onAppendPulse(double pulse)
{
	static std::deque<double> pulseHistory; // 用于存储最近五次脉冲值

	pulse = std::abs(pulse - lastPulse); // 计算当前脉冲与上次脉冲的绝对差值
	lastPulse = pulse; // 更新上次脉冲值

	// 将当前脉冲值加入历史记录
	pulseHistory.push_back(pulse);

	// 如果历史记录超过五次，移除最早的一次
	if (pulseHistory.size() > 5)
	{
		pulseHistory.pop_front();
	}

	// 计算最近五次脉冲的平均值
	double sum = std::accumulate(pulseHistory.begin(), pulseHistory.end(), 0.0);
	pulseAverage = (pulseHistory.empty()) ? 0.0 : (sum / pulseHistory.size());


}

void DetachUtiltyThreadSmartCroppingOfBags::onAppendPixel(double pixel)
{
	if (runningStatePixelParaChange)
	{
		pixelSum = pixel;
		pixelCount = 0;
		runningStatePixelParaChange = false;
	}
	else
	{
		pixelSum += pixel; 
		++pixelCount;

		pixelAverage = (pixelCount == 0) ? 0.0 : (pixelSum / pixelCount);
	}


}



