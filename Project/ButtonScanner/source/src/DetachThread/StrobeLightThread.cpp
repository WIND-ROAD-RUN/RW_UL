#include"StrobeLightThread.hpp"

#include "ButtonUtilty.h"
#include "rqw_CameraObjectZMotion.hpp"
StrobeLightThread::StrobeLightThread(QObject* parent)
	: QThread(parent), running(false) {
}

StrobeLightThread::~StrobeLightThread()
{
	stopThread();
	wait();
}

void StrobeLightThread::startThread()
{
	running = true;
	if (!isRunning()) {
		start();
	}
}

void StrobeLightThread::stopThread()
{
	running = false; 
}

void StrobeLightThread::run()
{
	auto& motion = zwy::scc::GlobalMotion::getInstance().motionPtr;
	while (running) {
		motion->SetIOOut(ControlLines::strobeLightOut, true);
		QThread::msleep(50);
		motion->SetIOOut(ControlLines::strobeLightOut, false);
		QThread::msleep(50);
	}
}
