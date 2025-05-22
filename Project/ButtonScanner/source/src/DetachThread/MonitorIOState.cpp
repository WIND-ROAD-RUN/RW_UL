#include"MonitorIOState.hpp"
#include <rqw_CameraObject.hpp>
#include <ButtonUtilty.h>

MonitorIOStateThread::MonitorIOStateThread(QObject* parent)
    : QThread(parent){

}

void MonitorIOStateThread::setRunning(bool running)
{
    QMutexLocker locker(&m_mutex);
    m_running = running;
    if (m_running)
        m_waitCond.wakeAll();
}

void MonitorIOStateThread::destroyThread()
{
    {
        QMutexLocker locker(&m_mutex);
        m_exit = true;
        m_waitCond.wakeAll();
    }
    wait();
}

void MonitorIOStateThread::run()
{
    while (true) {
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

void MonitorIOStateThread::monitorDIState()
{
    auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
    emit DIState(ControlLines::stopIn, motionPtr->GetIOIn(ControlLines::stopIn));
    emit DIState(ControlLines::startIn, motionPtr->GetIOIn(ControlLines::startIn));
    emit DIState(ControlLines::airWarnIN, motionPtr->GetIOIn(ControlLines::airWarnIN));
    emit DIState(ControlLines::shutdownComputerIn, motionPtr->GetIOIn(ControlLines::shutdownComputerIn));
    emit DIState(ControlLines::camer1In, motionPtr->GetIOIn(ControlLines::camer1In));
    emit DIState(ControlLines::camer2In, motionPtr->GetIOIn(ControlLines::camer2In));
    emit DIState(ControlLines::camer3In, motionPtr->GetIOIn(ControlLines::camer3In));
    emit DIState(ControlLines::camer4In, motionPtr->GetIOIn(ControlLines::camer4In));
}

void MonitorIOStateThread::monitorDOState()
{
    auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	emit DOState(ControlLines::warnRedOut, motionPtr->GetIOOut(ControlLines::warnRedOut));
    emit DOState(ControlLines::motoPowerOut, motionPtr->GetIOOut(ControlLines::motoPowerOut));
    emit DOState(ControlLines::beltAsis, motionPtr->GetIOOut(ControlLines::beltAsis));
    emit DOState(ControlLines::warnGreenOut, motionPtr->GetIOOut(ControlLines::warnGreenOut));
    emit DOState(ControlLines::warnRedOut, motionPtr->GetIOOut(ControlLines::warnRedOut));
    emit DOState(ControlLines::warnUpLightOut, motionPtr->GetIOOut(ControlLines::warnUpLightOut));
    emit DOState(ControlLines::warnSideLightOut, motionPtr->GetIOOut(ControlLines::warnSideLightOut));
    emit DOState(ControlLines::warnDownLightOut, motionPtr->GetIOOut(ControlLines::warnDownLightOut));
}
