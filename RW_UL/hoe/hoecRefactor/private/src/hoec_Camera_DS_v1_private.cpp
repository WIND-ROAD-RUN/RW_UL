#include"hoec_Camera_DS_v1_private.hpp"

#include"DVPCamera.h"
#include"dvpir.h"
#include"dvpParam.h"

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
		}

		bool Camera_DS::setGain(size_t value)
		{
		}

		bool Camera_DS::setTriggerMode(CameraTriggerMode mode)
		{
		}

		bool Camera_DS::setInTriggerLine(size_t lineIndex)
		{
		}

		size_t Camera_DS::getExposureTime(bool& isGet)
		{
		}

		size_t Camera_DS::getGain(bool& isGet)
		{
		}

		CameraTriggerMode Camera_DS::getMonitorMode(bool& isGet)
		{
		}

		size_t Camera_DS::getTriggerLine(bool& isGet)
		{
		}

		bool Camera_DS::setOutTriggerConfig(const OutTriggerConfig& config)
		{
		}

		bool Camera_DS::outTrigger()
		{
		}

		bool Camera_DS::outTrigger(bool isOpen)
		{
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
		}

		void Camera_DS_Passive::ImageCallBackFunc()
		{
		}
	}
}