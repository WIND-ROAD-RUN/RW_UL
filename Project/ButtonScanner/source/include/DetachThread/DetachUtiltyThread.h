#pragma once

#include <QThread>
#include <QDebug>
#include <atomic>
#include"rqw_LabelWarning.h"
#include"DlgWarn.h"

class DetachUtiltyThread : public QThread
{
	Q_OBJECT
private:
	bool isProcessing{false};
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
signals:
	void updateStatisticalInfo();
	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);
signals:
	void showDlgWarn(rw::rqw::WarningInfo info);

private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};