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
private:
	double getPulse(bool & isGet);
	double getAveragePulse(bool& isGet);
	double getAveragePulseBag(bool& isGet);
	double getAveragePixelBag(bool& isGet);
	double getLineHeight(bool& isGet);
signals:
	void updateCurrentPulse(double pulse);
	void updateMonitorRunningStateInfo(MonitorRunningStateInfo info);
private:
	std::atomic<bool> running; 
};


