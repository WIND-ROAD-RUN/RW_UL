#include"MonitorIOState.hpp"

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

}

void MonitorIOStateThread::monitorDOState()
{

}
