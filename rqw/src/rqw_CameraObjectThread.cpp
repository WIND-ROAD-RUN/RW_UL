#include"rqw_CameraObjectThread.hpp"

#include"rqw_CameraObject.hpp"


namespace rw
{
    namespace rqw
    {
        CameraPassiveThread::CameraPassiveThread(QObject* parent)
            :QThread(parent), _cameraObject(nullptr)
        {
        }

        CameraPassiveThread::~CameraPassiveThread()
        {
            stopMonitor();
            quit();
            wait();
            delete _cameraObject;
        }

        void CameraPassiveThread::initCamera(const rw::rqw::CameraMetaData& cameraMetaData,
            rw::rqw::CameraObjectTrigger triggerMode)
        {
            if (_cameraObject)
            {
                delete _cameraObject;
            }
            if (!_cameraObject)
            {
                _cameraObject = new rw::rqw::CameraPassiveObject();
                connect(_cameraObject, &rw::rqw::CameraPassiveObject::frameCaptured, this, &CameraPassiveThread::onFrameCaptured,Qt::DirectConnection);
                connect(_cameraObject, &rw::rqw::CameraPassiveObject::frameCapturedWithMetaData, this, &CameraPassiveThread::onFrameCapturedWithMetaData, Qt::DirectConnection);
                connect(_cameraObject, &rw::rqw::CameraPassiveObject::frameCapturedWithoutArgs, this, &CameraPassiveThread::onFrameCapturedWithoutArgs, Qt::DirectConnection);
                _cameraObject->initCamera(cameraMetaData, triggerMode);
            }
        }

        void CameraPassiveThread::startMonitor()
        {
            if (!this->isRunning())
            {
                //this->start();
            }
            if (_cameraObject)
            {
                _cameraObject->startMonitor();
            }
        }

        void CameraPassiveThread::stopMonitor()
        {
            if (_cameraObject)
            {
                _cameraObject->stopMonitor();
                delete _cameraObject;
                _cameraObject = nullptr;
            }
        }

        void CameraPassiveThread::setExposureTime(size_t value) const
        {
            if (_cameraObject)
            {
                _cameraObject->setExposureTime(value);
            }
        }

        void CameraPassiveThread::setGain(size_t value) const
        {
            if (_cameraObject)
            {
                _cameraObject->setGain(value);
            }
        }

        void CameraPassiveThread::setIOTime(size_t value) const
        {
            if (_cameraObject)
            {
                _cameraObject->setIOTime(value);
            }
        }

        void CameraPassiveThread::setTriggerMode(CameraObjectTrigger mode) const
        {
            if (_cameraObject)
            {
                _cameraObject->setTriggerMode(mode);
            }
        }

        void CameraPassiveThread::setTriggerLine(size_t lineIndex) const
        {
            if (_cameraObject)
            {
                _cameraObject->setTriggerLine(lineIndex);
            }
        }

        size_t CameraPassiveThread::getExposureTime() const
        {
            if (_cameraObject)
            {
                return _cameraObject->getExposureTime();
            }
            return 0;
        }

        size_t CameraPassiveThread::getGain() const
        {
            if (_cameraObject)
            {
                return _cameraObject->getGain();
            }
            return 0;
        }

        size_t CameraPassiveThread::getIOTime() const
        {
            if (_cameraObject)
            {
                return _cameraObject->getIOTime();
            }
            return 0;
        }

        CameraObjectTrigger CameraPassiveThread::getMonitorMode() const
        {
            if (_cameraObject)
            {
                return _cameraObject->getMonitorMode();
            }
            return CameraObjectTrigger::Undefined;
        }

        size_t CameraPassiveThread::getTriggerLine() const
        {
            if (_cameraObject)
            {
                return _cameraObject->getTriggerLine();
            }
            return 0;
        }

        void CameraPassiveThread::run()
        {
            exec();
        }

        void CameraPassiveThread::onFrameCaptured(cv::Mat frame)
        {
            emit frameCaptured(std::move(frame));
        }

        void CameraPassiveThread::onFrameCapturedWithMetaData(cv::Mat frame, rw::rqw::CameraMetaData cameraMetaData)
        {
            emit frameCapturedWithMetaData(std::move(frame), std::move(cameraMetaData));
        }

        void CameraPassiveThread::onFrameCapturedWithoutArgs()
        {
            emit frameCapturedWithoutArgs();
        }
    } // namespace rqw
} // namespace rw