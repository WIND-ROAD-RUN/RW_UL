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
public:
	rw::rqw::LabelWarning* warningLabel{nullptr};
private:
	unsigned long long olderWasteCount{};

protected:
	void run() override;
private:
	void CalculateRealtimeInformation(size_t s);
	void processWarningInfo(size_t s);
	void processOneWarn(rw::rqw::WarningInfo & info);
signals:
	void updateStatisticalInfo();
	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);
signals:
	void showDlgWarn(rw::rqw::WarningInfo info);

private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};