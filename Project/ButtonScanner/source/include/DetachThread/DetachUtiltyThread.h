#pragma once

#include <QThread>
#include <atomic>
#include"rqw_LabelWarning.h"

class DetachUtiltyThread : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{false};
	std::atomic_bool isProcessFinish{ false };
public:
	explicit DetachUtiltyThread(QObject* parent = nullptr);

	~DetachUtiltyThread() override;

	void startThread();

	void stopThread();
private:
	bool lastIsTakePictures{false};
private:
	bool lastIsShutDown{false};
	int shutdownCount{0};
private:
	unsigned long long lastWork1Count{ 0 };
	unsigned long long lastWork2Count{ 0 };
	unsigned long long lastWork3Count{ 0 };
	unsigned long long lastWork4Count{ 0 };
	bool isStopOnce{false};
public:
	rw::rqw::LabelWarning* warningLabel{nullptr};
private:
	unsigned long long olderWasteCount{};

protected:
	void run() override;
private:
	void CalculateRealtimeInformation(size_t s);
private:
	void processWarningInfo(size_t s);
	void processOneWarnGet(rw::rqw::WarningInfo & info);
	void processOneWarnFinsh(rw::rqw::WarningInfo& info);
	void openWarnAlarm(const rw::rqw::WarningInfo& info);
	void closeWarnAlarm(const rw::rqw::WarningInfo& info);
private:
	void processTrigger(size_t s);
private:
	void processTakePictures(size_t s);
private:
	void processShutdownIO(size_t s);
signals:
	void updateStatisticalInfo();
	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);
signals:
	void showDlgWarn(rw::rqw::WarningInfo info);
signals:
	void workTriggerError(int index);
signals:
	void closeTakePictures();
signals:
	void shutdownComputer(int time);
private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};
