#pragma once

#include <QThread>
#include <atomic>

#include<Utilty.hpp>

class DetachUtiltyThreadSmartCroppingOfBags : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{ false };
	std::atomic_bool isProcessFinish{ false };
public:
	explicit DetachUtiltyThreadSmartCroppingOfBags(QObject* parent = nullptr);

	~DetachUtiltyThreadSmartCroppingOfBags() override;

	void startThread();

	void stopThread();

protected:
	void run() override;
private:
	void getRunningState(size_t s);
	void getMainWindowRunningState(size_t s);
private:
	double getPulse(bool & isGet);
	double getAveragePulse(bool& isGet);
	double getAveragePulseBag(bool& isGet);
	double getAveragePixelBag(bool& isGet);
	double getLineHeight(bool& isGet);
signals:
	void updateMonitorRunningStateInfo(MonitorRunningStateInfo info);
	void updateMainWindowInfo(int i);
public slots:
	void onAppendPulse(double pulse);
	void onAppendPixel(double pixel);
	
private:
	std::atomic<bool> running;
	double lastPulse = 0.0;			// 上次脉冲值
	double pulseSum = 0.0;			// 累计和
	size_t pulseCount = 0;			// 累计计数
	double pulseAverage = 0.0;		// 脉冲平均值

	double lastPixel = 0.0;			// 上次像素值
	double pixelSum = 0.0;			// 累计和
	double pixelCount = 0;			// 累计计数
	double pixelAverage = 0.0;		// 像素平均值

	double daichangAverageFromPulse = 0.0;	// 根据平均脉冲求平均袋长
	double daichangAverageFromPixel = 0.0;	// 根据平均像素求平均袋长

	int lastDefectCount = 0;			// 上次剔废计数
};


