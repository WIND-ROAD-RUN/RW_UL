#pragma once

#include <QThread>
#include <QDebug>
#include <atomic>

class StatisticalInfoComputingThread : public QThread
{
	Q_OBJECT
public:
	explicit StatisticalInfoComputingThread(QObject* parent = nullptr);

	~StatisticalInfoComputingThread() override;

	void startThread();

	void stopThread();

protected:
	void run() override;

signals:
	void updateStatisticalInfo();
	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);

private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};