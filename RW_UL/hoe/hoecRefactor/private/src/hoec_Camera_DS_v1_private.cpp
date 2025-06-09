#include"hoec_Camera_DS_v1_private.hpp"



namespace rw
{
	namespace hoec_v1
	{
		Camera_DS::Camera_DS()
		{
		}

		Camera_DS::~Camera_DS()
		{
		}

		std::vector<std::string> Camera_DS::getCameraIpList()
		{
		}

		std::vector<CameraInfo> Camera_DS::getCameraInfoList()
		{
		}

		bool Camera_DS::initSDK()
		{
		}

		bool Camera_DS::unInitSDK()
		{
		}

		bool Camera_DS::connectCamera()
		{
		}

		bool Camera_DS::getConnectState(bool& isGet)
		{
		}

		bool Camera_DS::setFrameRate(float cameraFrameRate)
		{
		}

		float Camera_DS::getFrameRate(bool& isGet)
		{
		}

		bool Camera_DS::startMonitor()
		{
		}

		bool Camera_DS::stopMonitor()
		{
		}

		bool Camera_DS::setHeartbeatTime(size_t heartBeatTime)
		{
		}

		size_t Camera_DS::getHeartbeatTime(bool& isGet)
		{
		}

		bool Camera_DS::setExposureTime(size_t value)
		{
			dvpStatus status;

			status = dvpSetExposure(m_cameraHandle, value);

			if (status == DVP_STATUS_OK)
			{
				return true;
			}

			return false;
		}

		bool Camera_DS::setGain(size_t value)
		{
			dvpStatus status;

			status = dvpSetAnalogGain(m_cameraHandle, value);

			if (status == DVP_STATUS_OK)
			{
				return true;
			}

			return false;
		}

		bool Camera_DS::setTriggerMode(CameraTriggerMode mode)
		{
		}

		bool Camera_DS::setInTriggerLine(size_t lineIndex)
		{
		}

		size_t Camera_DS::getExposureTime(bool& isGet)
		{
			double exposure = 0.0;
			dvpStatus status = dvpGetExposure(m_cameraHandle, &exposure);
			if (status == DVP_STATUS_OK)
			{
				isGet = true;
				return exposure;
			}
			isGet = false;
			return 0;
		}

		size_t Camera_DS::getGain(bool& isGet)
		{
			float gain = 0.0;
			dvpStatus status = dvpGetAnalogGain(m_cameraHandle, &gain);
			if (status == DVP_STATUS_OK)
			{
				isGet = true;
				return gain;
			}
			isGet = false;
			return 0;
		}

		CameraTriggerMode Camera_DS::getMonitorMode(bool& isGet)
		{
			
		}

		size_t Camera_DS::getTriggerLine(bool& isGet)
		{
			dvpStatus status;

			dvpTriggerSource dvitriggersource;

			status = dvpGetTriggerSource(m_cameraHandle, &dvitriggersource);

			if (status == DVP_STATUS_OK)
			{
				isGet = true;
				return dvitriggersource;
			}
			isGet = false;
			return 0;
		}

		bool Camera_DS::setOutTriggerConfig(const OutTriggerConfig& config)
		{
			return false;

		}

		bool Camera_DS::outTrigger()
		{
			return false;

		}

		bool Camera_DS::outTrigger(bool isOpen)
		{
			return false;

		}

		bool Camera_DS::setPreDivider(size_t number)
		{
			dvpStatus status;

			if (!m_cameraHandle)
			{
				return false;

			}
			status = dvpSetConfigString(m_cameraHandle, "cLineTrigFreqPreDiv", std::to_string(number).c_str());
			if (status != DVP_STATUS_OK)
			{

				return false;
			}
			else
			{
				return true;


			}

		}

		bool Camera_DS::setMultiplier(size_t number)
		{
			dvpStatus status;

			if (!m_cameraHandle)
			{
				return false;

			}
			status = dvpSetConfigString(m_cameraHandle, "LineTrigFreqMult", std::to_string(number).c_str());
			if (status != DVP_STATUS_OK)
			{

				return false;
			}
			else
			{
				return true;


			}
		}

		bool Camera_DS::setPostDivider(size_t number)
		{
			dvpStatus status;

			if (!m_cameraHandle)
			{
				return false;

			}
			status = dvpSetConfigString(m_cameraHandle, "LineTrigFreqDiv", std::to_string(number).c_str());
			if (status != DVP_STATUS_OK)
			{

				return false;
			}
			else
			{
				return true;


			}
		}

		bool Camera_DS::getEncoderNumber(size_t& number)
		{
			char const* szValue;
			char const* szValue1;
			dvpStatus status;
			status = dvpGetConfigString(m_cameraHandle, "EncoderForwardCounter", &szValue);
			
			if (status != DVP_STATUS_OK)
			{
				return false;

			}
		
			status = dvpGetConfigString(m_cameraHandle, "EncoderBackwardCounter", &szValue1);

			if (status != DVP_STATUS_OK)
			{
				return false;

			}
			
			number = static_cast<size_t>(std::strtod(szValue, nullptr) - std::strtod(szValue1, nullptr));
			return true;
		}

		

		
		Camera_DS_Active::Camera_DS_Active()
		{
		}

		Camera_DS_Active::~Camera_DS_Active()
		{
		}

		cv::Mat Camera_DS_Active::getImage(bool& isGet)
		{
		}

		cv::Mat Camera_DS_Active::getImage()
		{
		}

		Camera_DS_Passive::Camera_DS_Passive(UserToCallBack userToCallback)
		{
		}

		Camera_DS_Passive::~Camera_DS_Passive()
		{
		}

		bool Camera_DS_Passive::RegisterCallBackFunc()
		{
			return false;

		}

		void Camera_DS_Passive::ImageCallBackFunc()
		{

		}
	}
}