#include "DetachUtiltyThread.h"

#include "GlobalStruct.hpp"
#include "rqw_CameraObjectZMotion.hpp"
#include <Utilty.hpp>

DetachUtiltyThreadZipper::DetachUtiltyThreadZipper(QObject* parent)
	: QThread(parent), running(false) {

}

DetachUtiltyThreadZipper::~DetachUtiltyThreadZipper()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void DetachUtiltyThreadZipper::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void DetachUtiltyThreadZipper::stopThread()
{
	running = false; // 停止线程
}

void DetachUtiltyThreadZipper::run()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& statisticalInfo = globalStruct.statisticalInfo;

	lastWork1Count = statisticalInfo.produceCount1.load();
	lastWork2Count = statisticalInfo.produceCount2.load();
	olderWasteCount = statisticalInfo.produceCount.load();

	static size_t s = 0;
	while (running) {
		QThread::sleep(1);
		CalculateRealtimeInformation(s);
		processWarningInfo(s);
		processTrigger(s);
		++s;
		if (s == 300)
		{
			s = 0;
		}
	}
}

void DetachUtiltyThreadZipper::CalculateRealtimeInformation(size_t s)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& statisticalInfo = globalStruct.statisticalInfo;
	if (s % 30 == 0)
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

void DetachUtiltyThreadZipper::processWarningInfo(size_t s)
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

void DetachUtiltyThreadZipper::processOneWarnGet(rw::rqw::WarningInfo& info)
{
	/*isProcessFinish = false;
	auto isEmpty = warningLabel->isEmptyWarningListThreadSafe();
	if (isEmpty)
	{
		return;
	}
	isProcessing = true;
	info = warningLabel->topWarningListThreadSafe();
	auto& config = GlobalStructDataZipper::getInstance().dlgWarningManagerConfig;
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
	}*/
}

void DetachUtiltyThreadZipper::processOneWarnFinsh(rw::rqw::WarningInfo& info)
{
	/*closeWarnAlarm(info);
	info = warningLabel->popWarningListThreadSafe();
	isProcessFinish = false;
	auto isEmpty = warningLabel->isEmptyWarningListThreadSafe();
	if (isEmpty)
	{
		return;
	}
	isProcessing = true;
	info = warningLabel->topWarningListThreadSafe();
	auto& config = GlobalStructDataZipper::getInstance().dlgWarningManagerConfig;
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
	}*/
}

void DetachUtiltyThreadZipper::openWarnAlarm(const rw::rqw::WarningInfo& info)
{
	auto& motion = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motion->SetIOOut(ControlLines::warnRedOut, true);
	motion->SetIOOut(ControlLines::warnGreenOut, false);
}

void DetachUtiltyThreadZipper::closeWarnAlarm(const rw::rqw::WarningInfo& info)
{
	auto& motion = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motion->SetIOOut(ControlLines::warnRedOut, false);
	motion->SetIOOut(ControlLines::warnGreenOut, true);
}

void DetachUtiltyThreadZipper::processTrigger(size_t s)
{
	if (s % 180 == 0)
	{
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& statisticalInfo = globalStruct.statisticalInfo;
		auto& runningState = globalStruct.runningState;
		bool isRun = runningState.load() == RunningState::OpenRemoveFunc;

		if (isRun)
		{
			if (isStopOnce)
			{
				isStopOnce = false;
				return;
			}
			auto newWork1Count = statisticalInfo.produceCount1.load();
			auto newWork2Count = statisticalInfo.produceCount2.load();

			if (newWork1Count == lastWork1Count)
			{
				emit workTriggerError(1);
			}
			if (newWork2Count == lastWork2Count)
			{
				emit workTriggerError(2);
			}

			lastWork1Count = newWork1Count;
			lastWork2Count = newWork2Count;
		}
		else
		{
			isStopOnce = true;
		}
	}
}
