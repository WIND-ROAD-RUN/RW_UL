#include "rqw_CameraObject.hpp"

#include"hoec_CameraFactory_v1.hpp"
#include"hoec_Camera_v1.hpp"

namespace rw
{
	namespace rqw
	{
		CameraPassiveObject::CameraPassiveObject(QObject* parent)
		{
		}

		CameraPassiveObject::~CameraPassiveObject()
			= default;

		bool CameraPassiveObject::startMonitor() const
		{
			return _cameraPassive->startMonitor();
		}

		bool CameraPassiveObject::stopMonitor() const
		{
			return _cameraPassive->stopMonitor();
		}

		bool CameraPassiveObject::setHeartbeatTime(size_t value) const
		{
			return _cameraPassive->setHeartbeatTime(value);
		}

		bool CameraPassiveObject::setFrameRate(float value) const
		{
			return _cameraPassive->setFrameRate(value);
		}

		size_t CameraPassiveObject::getHeartbeatTime(bool& isGet) const
		{
			return _cameraPassive->getHeartbeatTime(isGet);
		}

		float CameraPassiveObject::getFrameRate(bool& isGet) const
		{
			return _cameraPassive->getFrameRate(isGet);
		}

		bool CameraPassiveObject::setExposureTime(size_t value) const
		{
			return _cameraPassive->setExposureTime(value);
		}

		bool CameraPassiveObject::setGain(size_t value) const
		{
			return _cameraPassive->setGain(value);
		}

		bool CameraPassiveObject::setTriggerMode(CameraObjectTrigger mode) const
		{
			hoec_v1::CameraTriggerMode hoecTrigger;
			if (mode == CameraObjectTrigger::Hardware)
			{
				hoecTrigger = hoec_v1::CameraTriggerMode::HardwareTriggered;
			}
			else
			{
				hoecTrigger = hoec_v1::CameraTriggerMode::SoftwareTriggered;
			}
			return _cameraPassive->setTriggerMode(hoecTrigger);
		}

		bool CameraPassiveObject::setTriggerLine(size_t lineIndex) const
		{
			return _cameraPassive->setInTriggerLine(lineIndex);
		}

		size_t CameraPassiveObject::getExposureTime(bool& isGet) const
		{
			return _cameraPassive->getExposureTime(isGet);
		}

		size_t CameraPassiveObject::getGain(bool& isGet) const
		{
			return _cameraPassive->getGain(isGet);
		}


		CameraObjectTrigger CameraPassiveObject::getMonitorMode(bool& isGet) const
		{
			hoec_v1::CameraTriggerMode hoecTrigger = _cameraPassive->getMonitorMode(isGet);
			if (hoecTrigger == hoec_v1::CameraTriggerMode::HardwareTriggered)
			{
				return CameraObjectTrigger::Hardware;
			}
			else
			{
				return CameraObjectTrigger::Software;
			}
		}

		size_t CameraPassiveObject::getTriggerLine(bool& isGet) const
		{
			return _cameraPassive->getTriggerLine(isGet);
		}

		size_t CameraPassiveObject::getHeartbeatTime() const
		{
			bool isGet=false;
			return _cameraPassive->getHeartbeatTime(isGet);
		}

		float CameraPassiveObject::getFrameRate() const
		{
			bool isGet = false;
			return _cameraPassive->getFrameRate(isGet);
		}

		size_t CameraPassiveObject::getExposureTime() const
		{
			bool isGet = false;
			return _cameraPassive->getExposureTime(isGet);
		}

		size_t CameraPassiveObject::getGain() const
		{
			bool isGet = false;
			return _cameraPassive->getGain(isGet);
		}

		CameraObjectTrigger CameraPassiveObject::getMonitorMode() const
		{
			bool isGet = false;
			hoec_v1::CameraTriggerMode hoecTrigger = _cameraPassive->getMonitorMode(isGet);
			if (hoecTrigger == hoec_v1::CameraTriggerMode::HardwareTriggered)
			{
				return CameraObjectTrigger::Hardware;
			}
			else
			{
				return CameraObjectTrigger::Software;
			}
		}

		size_t CameraPassiveObject::getTriggerLine() const
		{
			bool isGet = false;
			return _cameraPassive->getTriggerLine(isGet);
		}

		void CameraPassiveObject::initCamera(const CameraMetaData& cameraMetaData, CameraObjectTrigger triggerMode)
		{
			_cameraMetaData = cameraMetaData;
			hoec_v1::CameraIP hoecCameraIp;
			hoecCameraIp.ip = cameraMetaData.ip.toStdString();
			hoecCameraIp.provider = hoec_v1::from_string(cameraMetaData.provider.toStdString());

			if (hoecCameraIp.provider==hoec_v1::CameraProvider::MVS)
			{
				hoec_v1::CameraTriggerMode hoecTrigger;
				if (triggerMode == CameraObjectTrigger::Hardware)
				{
					hoecTrigger = hoec_v1::CameraTriggerMode::HardwareTriggered;
				}
				else
				{
					hoecTrigger = hoec_v1::CameraTriggerMode::SoftwareTriggered;
				}

				_cameraPassive = hoec_v1::CameraFactory::CreatePassiveCamera(hoecCameraIp, hoecTrigger, [this](cv::Mat  mat)
					{
						emit frameCaptured(std::move(mat));
					});
			}
			else if (hoecCameraIp.provider == hoec_v1::CameraProvider::DS)
			{
				_cameraPassive = hoec_v1::CameraFactory::CreatePassiveCameraDS(hoecCameraIp, [this](cv::Mat  mat)
					{
						emit frameCaptured(std::move(mat));
					});
			}

			_cameraPassive->RegisterCallBackFunc();
		}
		bool CameraPassiveObject::getConnectState(bool& isGet)
		{
			if (_cameraPassive) {
				return _cameraPassive->getConnectState(isGet);
			}
			return false;
		}

		bool CameraPassiveObject::getConnectState()
		{
			if (_cameraPassive) {
				bool isGet = false;
				return _cameraPassive->getConnectState(isGet);
			}
			return false;
		}

		void CameraPassiveObject::setOutTriggerConfig(const OutTriggerConfig& config)
		{
			if (_cameraPassive) {
				rw::hoec_v1::OutTriggerConfig configHoec;
				configHoec.delayValue = config.delayValue;
				configHoec.strobeEnable = config.strobeEnable;
				configHoec.durationValue = config.durationValue;
				configHoec.lineMode = config.lineMode;
				configHoec.lineSelector = config.lineSelector;
				configHoec.lineSource = config.lineSource;
				configHoec.preDelayValue = config.preDelayValue;
				_cameraPassive->setOutTriggerConfig(configHoec);
			}
		}

		void CameraPassiveObject::outTrigger()
		{
			if (_cameraPassive) {
				_cameraPassive->outTrigger();
			}
		}

		void CameraPassiveObject::outTrigger(bool isOpen)
		{
			if (_cameraPassive) {
				_cameraPassive->outTrigger(isOpen);
			}
		}
	} // namespace rqw
} // namespace rw
