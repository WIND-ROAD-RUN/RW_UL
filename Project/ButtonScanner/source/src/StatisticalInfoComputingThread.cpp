#include "stdafx.h"
#include"StatisticalInfoComputingThread.h"

#include"GlobalStruct.h"

StatisticalInfoComputingThread::StatisticalInfoComputingThread(QObject* parent)
	: QThread(parent), running(false) {
}

StatisticalInfoComputingThread::~StatisticalInfoComputingThread()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void StatisticalInfoComputingThread::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void StatisticalInfoComputingThread::stopThread()
{
	running = false; // 停止线程
}

void StatisticalInfoComputingThread::run()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& statisticalInfo = globalStruct.statisticalInfo;
	unsigned long olderWasteCount = statisticalInfo.produceCount.load();
	while (running) {
		static size_t s;
		QThread::sleep(1);
		//每60s计算剔除功能
		if (s == 60)
		{
			auto newWasteCount = statisticalInfo.produceCount.load();
			auto rate = static_cast<double>(newWasteCount - olderWasteCount);
			//removeRate后使用为生产速度计算
			statisticalInfo.removeRate = rate;
			olderWasteCount = statisticalInfo.produceCount.load();
			s = 0;
		}

		// 计算生产良率
		auto totalCount = statisticalInfo.produceCount.load();
		auto wasteCount = statisticalInfo.wasteCount.load();
		if (totalCount == 0)
		{
			continue;
		}
		statisticalInfo.productionYield = (static_cast<double>(totalCount - wasteCount) / totalCount) * 100;

		// 发送信号更新UI
		emit updateStatisticalInfo();
		s++;
	}
}