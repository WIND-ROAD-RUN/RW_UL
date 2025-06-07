#include "stdafx.h"
#include"DetachUtiltyThread.h"

#include"GlobalStruct.h"
#include "rqw_CameraObjectZMotion.hpp"

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
	 lastWork1Count = statisticalInfo.produceCount1.load();
	 lastWork2Count = statisticalInfo.produceCount2.load();
	 lastWork3Count = statisticalInfo.produceCount3.load();
	 lastWork4Count = statisticalInfo.produceCount4.load();
	olderWasteCount = statisticalInfo.produceCount.load();
	static size_t s = 0;
	while (running) {
		QThread::sleep(1);
		CalculateRealtimeInformation(s);
		processWarningInfo(s);
		processTrigger(s);
		processTakePictures(s);
		processShutdownIO(s);
		++s;
		if (s==300)
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

void DetachUtiltyThread::processTrigger(size_t s)
{
	if (s%180==0)
	{
		auto& globalStruct = GlobalStructData::getInstance();
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
			auto newWork3Count = statisticalInfo.produceCount3.load();
			auto newWork4Count = statisticalInfo.produceCount4.load();

			if (newWork1Count== lastWork1Count)
			{
				emit workTriggerError(1);
			}
			if (newWork2Count == lastWork2Count)
			{
				emit workTriggerError(2);
			}
			if (newWork3Count == lastWork3Count)
			{
				emit workTriggerError(3);
			}
			if (newWork4Count == lastWork4Count)
			{
				emit workTriggerError(4);
			}


			lastWork1Count = newWork1Count;
			lastWork2Count = newWork2Count;
			lastWork3Count = newWork3Count;
			lastWork4Count = newWork4Count;
		}
		else
		{
			isStopOnce = true;
		}
	}
}

void DetachUtiltyThread::processTakePictures(size_t s)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& isTakePictures = globalStruct.mainWindowConfig.isTakePictures;
	auto& isSavePicturesLong = globalStruct.dlgProduceLineSetConfig.takePicturesLong;
	if (s%60==0)
	{
		if (lastIsTakePictures==true&& isTakePictures==true&& isSavePicturesLong==false)
		{
			emit closeTakePictures();
		}
		lastIsTakePictures = isTakePictures;
	}
}

void DetachUtiltyThread::processShutdownIO(size_t s)
{
	if (s%1==0)
	{
		auto& motion = zwy::scc::GlobalMotion::getInstance().motionPtr;
		auto isShutdown = motion->GetIOIn(ControlLines::shutdownComputerIn);

		if (lastIsShutDown)
		{
			shutdownCount++;
			emit shutdownComputer(shutdownCount);
		}
		else
		{
			if (isShutdown)
			{
				emit shutdownComputer(shutdownCount);
			}
			else
			{
				shutdownCount = 0;
				emit shutdownComputer(-1);
			}
		}
		lastIsShutDown = isShutdown;
	}
}
