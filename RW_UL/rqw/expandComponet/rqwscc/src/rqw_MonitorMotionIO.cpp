#include"rqw_MonitorMotionIO.hpp"

namespace rw
{
    namespace rqw
    {
        MonitorZMotionIOStateThread::MonitorZMotionIOStateThread(QObject* parent)
            : QThread(parent) {

        }

        void MonitorZMotionIOStateThread::startRunning(bool running)
        {
            QMutexLocker locker(&m_mutex);
            m_running = running;
            if (m_running)
                m_waitCond.wakeAll();
        }

        void MonitorZMotionIOStateThread::destroyThread()
        {
            {
                QMutexLocker locker(&m_mutex);
                m_exit = true;
                m_waitCond.wakeAll();
            }
            wait();
        }

        void MonitorZMotionIOStateThread::setMonitorFrequency(unsigned long ms)
        {
            _monitorFrequency = ms;
        }

        void MonitorZMotionIOStateThread::setMonitorObject(ZMotion& zMotion)
        {
            _monitorObject = &zMotion;
        }

        void MonitorZMotionIOStateThread::setMonitorIList(const QVector<size_t>& monitorIList)
        {
            _monitorIList = monitorIList;
        }

        void MonitorZMotionIOStateThread::setMonitorOList(const QVector<size_t>& monitorOList)
        {
            _monitorOList = monitorOList;
        }

        void MonitorZMotionIOStateThread::setMonitorIOList(const QVector<size_t>& monitorIList,
	        const QVector<size_t>& monitorOList)
        {
            _monitorIList = monitorIList;
            _monitorOList = monitorOList;
        }


        void MonitorZMotionIOStateThread::run()
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

                msleep(_monitorFrequency);
            }
        }

        void MonitorZMotionIOStateThread::monitorDIState()
        {
            if (!_monitorObject)
            {
                return;
            }

            for (const auto & item: _monitorIList)
            {
                emit DIState(item, _monitorObject->getIOIn(item));

            }
        }

        void MonitorZMotionIOStateThread::monitorDOState()
        {
            if (!_monitorObject)
            {
                return;
            }

        	for (const auto& item : _monitorIList)
            {
                emit DIState(item, _monitorObject->getIOIn(item));

            }
        }
    }
}




