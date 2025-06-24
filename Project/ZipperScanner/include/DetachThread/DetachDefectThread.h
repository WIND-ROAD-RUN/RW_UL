#pragma once

#include <atomic>
#include<QThread>

#include"rqw_LabelWarning.h"
#include"dsl_PriorityQueue.hpp"

#include"Utilty.hpp"
#include"GlobalStruct.hpp"


class DetachDefectThreadZipper : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{ false };
	std::atomic_bool isProcessFinish{ false };
public:
	explicit DetachDefectThreadZipper(QObject* parent = nullptr);

	~DetachDefectThreadZipper() override;

	void startThread();

	void stopThread();

	void processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<DefectValueInfo, DefectValueInfo>> &queue);
	void processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<DefectValueInfo, DefectValueInfo>> &queue);

signals:
	void findIsBad(size_t index);
protected:
	void run() override;
private:
	std::atomic<bool> running; // 使用原子变量保证线程安全

	float preLocation{ 0.0f };
};
