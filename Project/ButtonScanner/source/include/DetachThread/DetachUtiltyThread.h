#pragma once

#include <QThread>
#include <QDebug>
#include <atomic>

class DetachUtiltyThread : public QThread
{
	Q_OBJECT
public:
	explicit DetachUtiltyThread(QObject* parent = nullptr);

	~DetachUtiltyThread() override;

	void startThread();

	void stopThread();
private:
	unsigned long long olderWasteCount{};

protected:
	void run() override;
private:
	void CalculateRealtimeInformation(int s);
signals:

	void updateStatisticalInfo();
	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);

private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};