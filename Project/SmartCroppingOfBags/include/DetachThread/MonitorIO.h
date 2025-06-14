#pragma once

#include <atomic>
#include <QThread>

#include"Utilty.hpp"
#include"GlobalStruct.hpp"

class MonitorIOSmartCroppingOfBags : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{ false };
	std::atomic_bool isProcessFinish{ false };

	// IO点位状态
	bool state = false;
	// 脉冲位置
	double location = 0;
public:
	explicit MonitorIOSmartCroppingOfBags(QObject* parent = nullptr);

	~MonitorIOSmartCroppingOfBags() override;

	void startThread();

	void stopThread();

signals:
	void findIsBad(size_t index);
protected:
	void run() override;
private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};