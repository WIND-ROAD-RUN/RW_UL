#pragma once

#include <atomic>
#include <QThread>

#include "dsl_PriorityQueue.hpp"
#include"Utilty.hpp"
#include"GlobalStruct.hpp"

class DetachDefectThreadSmartCroppingOfBags : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{ false };
	std::atomic_bool isProcessFinish{ false };
public:
	explicit DetachDefectThreadSmartCroppingOfBags(QObject* parent = nullptr);

	~DetachDefectThreadSmartCroppingOfBags() override;

	void startThread();

	void stopThread();

	void processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<double, double>>& queue);
	void processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<double, double>>& queue);

signals:
	void findIsBad(size_t index);
protected:
	void run() override;
private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};