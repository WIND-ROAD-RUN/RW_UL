#include"rqw_CameraObjectThread.hpp"

#include"rqw_CameraObject.hpp"
#include"hoec_CameraException.hpp"

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
			quit();
			wait();
			if (_cameraObject->getConnectState()) {
				if (_cameraObject) {
					delete _cameraObject;
				}
			}
			else {
				try
				{
					if (_cameraObject) {
						delete _cameraObject;
					}
				}
				catch (const std::exception&)
				{
				}
			}
		}

		void CameraPassiveThread::initCamera(const rw::rqw::CameraMetaData& cameraMetaData,
			rw::rqw::CameraObjectTrigger triggerMode, size_t motionInde)
		{
			if (_cameraObject)
			{
				delete _cameraObject;
			}
			if (!_cameraObject)
			{
				_cameraObject = new rw::rqw::CameraPassiveObject();
				connect(_cameraObject, &rw::rqw::CameraPassiveObject::frameCaptured, this, &CameraPassiveThread::onFrameCaptured, Qt::DirectConnection);
				_cameraObject->motionInde = motionInde;
				_cameraObject->initCamera(cameraMetaData, triggerMode);
			}
		}

		bool CameraPassiveThread::getConnectState()
		{
			return _cameraObject->getConnectState();
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

		void CameraPassiveThread::setHeartbeatTime(size_t value) const
		{
			if (_cameraObject)
			{
				_cameraObject->setHeartbeatTime(value);
			}
		}

		void CameraPassiveThread::setFrameRate(float value) const
		{
			if (_cameraObject)
			{
				_cameraObject->setFrameRate(value);
			}
		}

		size_t CameraPassiveThread::getHeartbeatTime() const
		{
			if (_cameraObject)
			{
				return _cameraObject->getHeartbeatTime();
			}
			return 0;
		}

		float CameraPassiveThread::getFrameRate() const
		{
			if (_cameraObject)
			{
				return _cameraObject->getFrameRate();
			}
			return 0;
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

		void CameraPassiveThread::onFrameCaptured(cv::Mat frame, float location)
		{
			emit frameCaptured(std::move(frame), location, cameraIndex);
		}
	} // namespace rqw
} // namespace rw