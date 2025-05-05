#include"stdafx.h"

#include"MonitorCameraAndCardStateThread.h"
#include"GlobalStruct.h"
#include"scc_motion.h"
#include"hoec_Camera.hpp"
#include"rqw_CameraObjectThread.hpp"
#include"rqw_CameraObject.hpp"

size_t MonitorCameraAndCardStateThread::runtimeCounts = 0;

MonitorCameraAndCardStateThread::MonitorCameraAndCardStateThread(QObject* parent)
	: QThread(parent), running(false) {
}

MonitorCameraAndCardStateThread::~MonitorCameraAndCardStateThread()
{
	stopThread();
	wait();
}

void MonitorCameraAndCardStateThread::startThread()
{
	running = true;
	if (!isRunning()) {
		start();
	}
}

void MonitorCameraAndCardStateThread::stopThread()
{
	running = false;
}

void MonitorCameraAndCardStateThread::run()
{
	while (running) {
		QThread::msleep(2000);
		check_cardState();
		check_cameraState();
		runtimeCounts++;
		if (runtimeCounts == 4) {
			runtimeCounts = 0;
		}
	}
}

void MonitorCameraAndCardStateThread::check_cameraState()
{
	check_cameraState1();
	check_cameraState2();
	check_cameraState3();
	check_cameraState4();
}

void MonitorCameraAndCardStateThread::check_cameraState1()
{
	static bool isUpdateSate = false;

	auto& globalStruct = GlobalStructData::getInstance();

	if (runtimeCounts != 0) {
		return;
	}
	if (globalStruct.camera1) {
		if (globalStruct.camera1->getConnectState()) {
			if (!isUpdateSate) {
				emit updateCameraLabelState(1, true);
				isUpdateSate = true;
			}
		}
		else {
			emit destroyCamera1();
			emit updateCameraLabelState(1, false);
			emit addWarningInfo("相机1断连", true, 5000);
		}
	}
	else {
		emit buildCamera1();
		emit startMonitor1();
		emit updateCameraLabelState(1, false);
		isUpdateSate = false;
	}
}

void MonitorCameraAndCardStateThread::check_cameraState2()
{
	static bool isUpdateSate = false;

	auto& globalStruct = GlobalStructData::getInstance();

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
			emit addWarningInfo("相机2断连", true, 5000);
		}
	}
	else {
		emit buildCamera2();
		emit startMonitor2();
		emit updateCameraLabelState(2, false);
		isUpdateSate = false;
	}
}

void MonitorCameraAndCardStateThread::check_cameraState3()
{
	static bool isUpdateSate = false;

	auto& globalStruct = GlobalStructData::getInstance();

	if (runtimeCounts != 2) {
		return;
	}

	if (globalStruct.camera3) {
		if (globalStruct.camera3->getConnectState()) {
			if (!isUpdateSate) {
				emit updateCameraLabelState(3, true);
				isUpdateSate = true;
			}
		}
		else {
			emit destroyCamera3();
			emit updateCameraLabelState(3, false);
			emit addWarningInfo("相机3断连", true, 5000);
		}
	}
	else {
		emit buildCamera3();
		emit startMonitor3();
		emit updateCameraLabelState(3, false);
		isUpdateSate = false;
	}
}

void MonitorCameraAndCardStateThread::check_cameraState4()
{
	static bool isUpdateSate = false;

	if (runtimeCounts != 3) {
		return;
	}

	auto& globalStruct = GlobalStructData::getInstance();
	if (globalStruct.camera4) {
		if (globalStruct.camera4->getConnectState()) {
			if (!isUpdateSate) {
				emit updateCameraLabelState(4, true);
				isUpdateSate = true;
			}
		}
		else {
			emit destroyCamera4();
			emit updateCameraLabelState(4, false);
			emit addWarningInfo("相机4断连", true, 5000);
		}
	}
	else {
		emit buildCamera4();
		emit startMonitor4();
		emit updateCameraLabelState(4, false);
		isUpdateSate = false;
	}
}

void MonitorCameraAndCardStateThread::check_cardState()
{
	//auto& globalStruct = GlobalStructData::getInstance();

	//auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;

	//bool  boardState = motionPtr.get()->getBoardState();
	//if (boardState == false)
	//{
	//    auto openRusult = motionPtr.get()->OpenBoard((char*)"192.168.0.11");
	//    if (openRusult) {
	//        emit updateCardLabelState(true);
	//    }
	//    else {
	//        emit updateCardLabelState(false);
	//    }
	//}
	//else
	//{
	//    emit updateCardLabelState(true);
	//}
}