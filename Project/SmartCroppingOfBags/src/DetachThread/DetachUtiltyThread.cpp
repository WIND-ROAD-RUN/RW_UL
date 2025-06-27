#include "DetachUtiltyThread.h"

#include "GlobalStruct.hpp"
#include "scc_motion.h"



DetachUtiltyThreadSmartCroppingOfBags::DetachUtiltyThreadSmartCroppingOfBags(QObject* parent)
	: QThread(parent), running(false) {
	/*rw::ModelEngineConfig config;
	config.conf_threshold = 0.1f;
	config.nms_threshold = 0.1f;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::LetterBox;
	config.letterBoxColor = cv::Scalar(114, 114, 114);
	config.modelPath = globalPath.modelPath.toStdString();
	engine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::Yolov11_Det, rw::ModelEngineDeployType::TensorRT);*/
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
		cv::Mat mat = cv::Mat::zeros(1024, 1024, CV_8UC3);
		//engine->processImg(mat);
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
		if (GlobalStructThreadSmartCroppingOfBags::getInstance()._isUpdateMonitoyInfo)
		{
			emit updateMonitorRunningStateInfo(info);
		}
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

	double pulseTemp=pulse;
	pulse = std::abs(pulse - lastPulse); // 计算当前脉冲与上次脉冲的绝对差值
	lastPulse = pulseTemp; // 更新上次脉冲值

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
	static std::deque<double> pixelHistory; // 用于存储最近五次像素值

	// 将当前像素值加入历史记录
	pixelHistory.push_back(pixel);

	// 如果历史记录超过五次，移除最早的一次
	if (pixelHistory.size() > 5) {
		pixelHistory.pop_front();
	}

	// 计算最近五次像素的平均值
	double sum = std::accumulate(pixelHistory.begin(), pixelHistory.end(), 0.0);
	pixelAverage = (pixelHistory.empty()) ? 0.0 : (sum / pixelHistory.size());
}



