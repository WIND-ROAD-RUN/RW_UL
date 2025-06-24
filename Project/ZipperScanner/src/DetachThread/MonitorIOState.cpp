#include "MonitorIOState.h"

#include "rqw_CameraObjectZMotion.hpp"
#include "Utilty.hpp"
#include "GlobalStruct.hpp"


MonitorIOStateThreadZipper::MonitorIOStateThreadZipper(QObject* parent)
	: QThread(parent) {

}

void MonitorIOStateThreadZipper::setRunning(bool running)
{
	QMutexLocker locker(&m_mutex);
	m_running = running;
	if (m_running)
		m_waitCond.wakeAll();
}

void MonitorIOStateThreadZipper::destroyThread()
{
    {
        QMutexLocker locker(&m_mutex);
        m_exit = true;
        m_waitCond.wakeAll();
    }
    wait();
}

void MonitorIOStateThreadZipper::run()
{
	auto& isStart = GlobalStructDataZipper::getInstance()._isUpdateMonitorInfo;
    while (isStart) {
        m_mutex.lock();
        if (m_exit) {
            m_mutex.unlock();
            break;
        }
        if (!m_running) {
            m_waitCond.wait(&m_mutex);
            if (m_exit) {
                m_mutex.unlock();
                break;
            }
        }
        m_mutex.unlock();

        monitorDIState();
        monitorDOState();

        msleep(100);
    }
}

void MonitorIOStateThreadZipper::monitorDIState()
{
    auto& zmotion = GlobalStructDataZipper::getInstance().zmotion;
    emit DIState(ControlLines::qidonganniuIn, zmotion.getIOIn(ControlLines::qidonganniuIn));
	emit DIState(ControlLines::jitingIn, zmotion.getIOIn(ControlLines::jitingIn));
	emit DIState(ControlLines::lalianlawanIn, zmotion.getIOIn(ControlLines::lalianlawanIn));
}

void MonitorIOStateThreadZipper::monitorDOState()
{
	auto& zmotion = GlobalStructDataZipper::getInstance().zmotion;
	emit DOState(ControlLines::bujindianjimaichongOut, zmotion.getIOOut(ControlLines::bujindianjimaichongOut));
	emit DOState(ControlLines::chongkongOUT, zmotion.getIOOut(ControlLines::chongkongOUT));
	emit DOState(ControlLines::tuojiOut, zmotion.getIOOut(ControlLines::tuojiOut));
}
