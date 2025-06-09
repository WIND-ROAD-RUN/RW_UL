#pragma once

#include <QThread>
#include <atomic>

#include "dsl_PriorityQueue.hpp"
#include "GlobalStruct.hpp"
#include"rqw_LabelWarning.h"


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

	void processQueue1(std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > &queue);
	void processQueue2(std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > &queue);

signals:
	void findIsBad(size_t index);
protected:
	void run() override;
private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};
