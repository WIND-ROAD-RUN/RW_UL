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
			rw::rqw::CameraObjectTrigger triggerMode)
		{
			if (_cameraObject)
			{
				delete _cameraObject;
			}
			if (!_cameraObject)
			{
				_cameraObject = new rw::rqw::CameraPassiveObject();
				connect(_cameraObject, &rw::rqw::CameraPassiveObject::frameCaptured, this, &CameraPassiveThread::onFrameCaptured, Qt::DirectConnection);
				_cameraObject->initCamera(cameraMetaData, triggerMode);
			}
		}

		bool CameraPassiveThread::getConnectState()
		{
			return _cameraObject->getConnectState();
		}

		bool CameraPassiveThread::startMonitor()
		{
			if (!this->isRunning())
			{
				//this->start();
			}
			if (_cameraObject)
			{
				return _cameraObject->startMonitor();
			}
		}

		bool CameraPassiveThread::stopMonitor()
		{
			if (_cameraObject)
			{
				return _cameraObject->stopMonitor();
				delete _cameraObject;
				_cameraObject = nullptr;
			}
		}

		bool CameraPassiveThread::setHeartbeatTime(size_t value) const
		{
			if (_cameraObject)
			{
				return _cameraObject->setHeartbeatTime(value);
			}
		}

		bool CameraPassiveThread::setFrameRate(float value) const
		{
			if (_cameraObject)
			{
				return _cameraObject->setFrameRate(value);
			}
			return false;
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

		size_t CameraPassiveThread::getExposureTime(bool& isGet) const
		{
			if (_cameraObject)
			{
				return _cameraObject->getExposureTime(isGet);
			}
			return 0;
		}

		size_t CameraPassiveThread::getGain(bool& isGet) const
		{
			if (_cameraObject)
			{
				return _cameraObject->getGain(isGet);
			}
			return 0;
		}

		CameraObjectTrigger CameraPassiveThread::getMonitorMode(bool& isGet) const
		{
			if (_cameraObject)
			{
				return _cameraObject->getMonitorMode(isGet);
			}
			return CameraObjectTrigger::Undefined;
		}

		size_t CameraPassiveThread::getTriggerLine(bool& isGet) const
		{
			if (_cameraObject)
			{
				return _cameraObject->getTriggerLine(isGet);
			}
			return 0;
		}

		size_t CameraPassiveThread::getHeartbeatTime(bool& isGet) const
		{
			if (_cameraObject)
			{
				return _cameraObject->getHeartbeatTime(isGet);
			}
			return 0;
		}

		float CameraPassiveThread::getFrameRate(bool& isGet) const
		{
			if (_cameraObject)
			{
				return _cameraObject->getFrameRate(isGet);
			}
			return 0;
		}

		bool CameraPassiveThread::setExposureTime(size_t value) const
		{
			if (_cameraObject)
			{
				return _cameraObject->setExposureTime(value);
			}
			return false;
		}

		bool CameraPassiveThread::setGain(size_t value) const
		{
			if (_cameraObject)
			{
				return _cameraObject->setGain(value);
			}
			return false;
		}


		bool CameraPassiveThread::setTriggerMode(CameraObjectTrigger mode) const
		{
			if (_cameraObject)
			{
				return _cameraObject->setTriggerMode(mode);
			}
			return false;
		}

		bool CameraPassiveThread::setTriggerLine(size_t lineIndex) const
		{
			if (_cameraObject)
			{
				return _cameraObject->setTriggerLine(lineIndex);
			}
			return false;
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
			emit frameCaptured(std::move(frame), cameraIndex);
		}

		bool CameraPassiveThread::setOutTriggerConfig(const OutTriggerConfig& config)
		{
			if (_cameraObject)
			{
				 _cameraObject->setOutTriggerConfig(config);
			}
			return false;
		}

		bool CameraPassiveThread::outTrigger()
		{
			if (_cameraObject)
			{
				_cameraObject->outTrigger();
			}
			return false;
		}

		bool CameraPassiveThread::outTrigger(bool isOpen)
		{
			if (_cameraObject)
			{
				_cameraObject->outTrigger(isOpen);
			}
			return false;
		}
	} // namespace rqw
} // namespace rw