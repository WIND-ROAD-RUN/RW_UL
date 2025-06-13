#include "CameraAndCardStateThread.h"

#include "GlobalStruct.hpp"
#include "rqw_CameraObjectZMotion.hpp"

size_t CameraAndCardStateThreadZipper::runtimeCounts=0;

CameraAndCardStateThreadZipper::CameraAndCardStateThreadZipper(QObject* parent)
	: QThread(parent), running(false), _dlgProductSet(GlobalStructDataZipper::getInstance().setConfig){
}

CameraAndCardStateThreadZipper::~CameraAndCardStateThreadZipper()
{
	stopThread();
	wait();
}

void CameraAndCardStateThreadZipper::startThread()
{
	running = true;
	if (!isRunning()) {
		start();
	}
}

void CameraAndCardStateThreadZipper::stopThread()
{
	running = false;
}

void CameraAndCardStateThreadZipper::run()
{
	while (running) {
		QThread::msleep(2000);
		if (_dlgProductSet.yundongkongzhiqichonglian)
		{
			check_cardState();
		}
			check_cameraState();
		runtimeCounts++;
		if (runtimeCounts == 4) {
			runtimeCounts = 0;
		}
	}
}

void CameraAndCardStateThreadZipper::check_cameraState()
{
	check_cameraState1();
	if (_dlgProductSet.qiyongerxiangji)
	{
		check_cameraState2();
	}
}

void CameraAndCardStateThreadZipper::check_cameraState1()
{
	static bool isUpdateState = false;

	auto& globalStruct = GlobalStructDataZipper::getInstance();

	if (runtimeCounts != 0) {
		return;
	}
	if (globalStruct.camera1) {
		if (globalStruct.camera1->getConnectState()) {
			if (!isUpdateState) {
				emit updateCameraLabelState(1, true);
				isUpdateState = true;
			}
		}
		else {
			emit destroyCamera1();
			emit updateCameraLabelState(1, false);
			//emit addWarningInfo("相机1断连", true, 5000);
		}
	}
	else {
		emit buildCamera1();
		//emit startMonitor1();
		emit updateCameraLabelState(1, false);
		isUpdateState = false;
	}
}

void CameraAndCardStateThreadZipper::check_cameraState2()
{
	static bool isUpdateSate = false;

	auto& globalStruct = GlobalStructDataZipper::getInstance();

	if (runtimeCounts != 1) {
		return;
	}

	if (globalStruct.camera2) {
		if (globalStruct.camera2->getConnectState()) {
			if (!isUpdateSate) {
				emit updateCameraLabelState(2, true);
				isUpdateSate = true;
			}
		}
		else {
			emit destroyCamera2();
			emit updateCameraLabelState(2, false);
			//emit addWarningInfo("相机2断连", true, 5000);
		}
	}
	else {
		emit buildCamera2();
		//emit startMonitor2();
		emit updateCameraLabelState(2, false);
		isUpdateSate = false;
	}
}

void CameraAndCardStateThreadZipper::check_cardState()
{
	/*auto& globalStruct = GlobalStructDataZipper::getInstance();

	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;

	bool  boardState = motionPtr.get()->getBoardState();
	if (boardState == false)
	{
	    auto openRusult = motionPtr.get()->OpenBoard((char*)"192.168.0.11");
	    if (openRusult) {
	        emit updateCardLabelState(true);
	    }
	    else {
	        emit updateCardLabelState(false);
	    }
	}
	else
	{
	    emit updateCardLabelState(true);
	}*/
}
