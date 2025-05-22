#include "stdafx.h"
#include"DetachUtiltyThread.h"

#include"GlobalStruct.h"
#include "rqw_CameraObject.hpp"

DetachUtiltyThread::DetachUtiltyThread(QObject* parent)
	: QThread(parent), running(false) {
}

DetachUtiltyThread::~DetachUtiltyThread()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void DetachUtiltyThread::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void DetachUtiltyThread::stopThread()
{
	running = false; // 停止线程
}

void DetachUtiltyThread::run()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& statisticalInfo = globalStruct.statisticalInfo;
	olderWasteCount = statisticalInfo.produceCount.load();
	static size_t s = 0;
	while (running) {
		QThread::sleep(1);
		CalculateRealtimeInformation(s);
		processWarningInfo(s);
		++s;
		if (s==60)
		{
			s = 0;
		}
	}
}

void DetachUtiltyThread::CalculateRealtimeInformation(size_t s)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& statisticalInfo = globalStruct.statisticalInfo;
	if (s % 30==0)
	{
		auto newWasteCount = statisticalInfo.produceCount.load();
		long long rate = newWasteCount - olderWasteCount;
		if (rate > 0)
		{
			//removeRate后使用为生产速度计算
			statisticalInfo.removeRate = rate * 2;
			olderWasteCount = statisticalInfo.produceCount.load();
		}
		else {
			olderWasteCount = newWasteCount;
		}
	}

	// 计算生产良率
	auto totalCount = statisticalInfo.produceCount.load();
	auto wasteCount = statisticalInfo.wasteCount.load();
	if (totalCount != 0)
	{
		if (totalCount > wasteCount)
		{
			statisticalInfo.productionYield = (static_cast<double>(totalCount - wasteCount) / totalCount) * 100;
		}
	}

	emit updateStatisticalInfo();
}

void DetachUtiltyThread::processWarningInfo(size_t s)
{
	static rw::rqw::WarningInfo warningInfo;
	if (isProcessFinish)
	{
		isProcessFinish = false;
		processOneWarnFinsh(warningInfo);
	}
	if (s % 2 == 0 && !isProcessing)
	{
		processOneWarnGet(warningInfo);
	}
}

void DetachUtiltyThread::processOneWarnGet(rw::rqw::WarningInfo& info)
{
	isProcessFinish = false;
	auto isEmpty = warningLabel->isEmptyWarningListThreadSafe();
	if (isEmpty)
	{
		return;
	}
	isProcessing = true;
	info = warningLabel->topWarningListThreadSafe();
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	auto isOpenWarn = config.findIsOpen(info.warningId);
	if (isOpenWarn)
	{
		emit showDlgWarn(info);
		openWarnAlarm(info);;
	}
	else
	{
		isProcessing = false;
		warningLabel->popWarningListThreadSafe();
	}
}

void DetachUtiltyThread::processOneWarnFinsh(rw::rqw::WarningInfo& info)
{
	closeWarnAlarm(info);
	info = warningLabel->popWarningListThreadSafe();
	isProcessFinish = false;
	auto isEmpty = warningLabel->isEmptyWarningListThreadSafe();
	if (isEmpty)
	{
		return;
	}
	isProcessing = true;
	info = warningLabel->topWarningListThreadSafe();
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	auto isOpenWarn = config.findIsOpen(info.warningId);
	if (isOpenWarn)
	{
		emit showDlgWarn(info);
		openWarnAlarm(info);
	}
	else
	{
		isProcessing = false;
		warningLabel->popWarningListThreadSafe();
	}

}

void DetachUtiltyThread::openWarnAlarm(const rw::rqw::WarningInfo& info)
{
	auto &motion=zwy::scc::GlobalMotion::getInstance().motionPtr;
	motion->SetIOOut(ControlLines::warnRedOut, true);
	motion->SetIOOut(ControlLines::warnGreenOut, false);
}

void DetachUtiltyThread::closeWarnAlarm(const rw::rqw::WarningInfo& info)
{
	auto& motion = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motion->SetIOOut(ControlLines::warnRedOut, false);
	motion->SetIOOut(ControlLines::warnGreenOut, true);
}
