#pragma once

#include <QThread>
#include <atomic>

class StrobeLightThread : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{ false };
	std::atomic_bool isProcessFinish{ false };
public:
	explicit StrobeLightThread(QObject* parent = nullptr);

	~StrobeLightThread() override;

	void startThread();

	void stopThread();

protected:
	void run() override;

private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};