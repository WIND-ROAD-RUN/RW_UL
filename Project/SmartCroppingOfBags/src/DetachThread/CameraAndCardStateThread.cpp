#include "CameraAndCardStateThread.h"

#include "GlobalStruct.hpp"

size_t CameraAndCardStateThreadSCOB::runtimeCounts;

CameraAndCardStateThreadSCOB::CameraAndCardStateThreadSCOB(QObject* parent)
	: QThread(parent), running(false), _dlgProductSet(GlobalStructDataSmartCroppingOfBags::getInstance().setConfig) {

}

CameraAndCardStateThreadSCOB::~CameraAndCardStateThreadSCOB()
{
	stopThread();
	wait();
}

void CameraAndCardStateThreadSCOB::startThread()
{
	running = true;
	if (!isRunning()) {
		start();
	}
}

void CameraAndCardStateThreadSCOB::stopThread()
{
	running = false;
}

void CameraAndCardStateThreadSCOB::run()
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

void CameraAndCardStateThreadSCOB::check_cameraState()
{
	check_cameraState1();
	if (_dlgProductSet.qiyonger)
	{
		check_cameraState2();
	}
}

void CameraAndCardStateThreadSCOB::check_cameraState1()
{
	static bool isUpdateState = false;

	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

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

void CameraAndCardStateThreadSCOB::check_cameraState2()
{
	static bool isUpdateSate = false;

	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

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

void CameraAndCardStateThreadSCOB::check_cardState()
{

}
