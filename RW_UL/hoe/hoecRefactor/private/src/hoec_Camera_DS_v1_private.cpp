#include"hoec_Camera_DS_v1_private.hpp"

#include"hoec_Camera_core_v1_private.hpp"

namespace rw
{
	namespace hoec_v1
	{
		Camera_DS::Camera_DS()
		{
		}

		Camera_DS::~Camera_DS()
		{
			if (!m_cameraHandle) {
				return;
			}

			if (_isMonitor) {
				dvpStop(m_cameraHandle);
				_isMonitor = false;
			}

			dvpClose(m_cameraHandle);
			m_cameraHandle = 0;
		}

		std::vector<std::string> Camera_DS::getCameraIpList()
		{
			std::vector<std::string> ipList;
			dvpUint32 count = 0;
			if (dvpRefresh(&count) == DVP_STATUS_OK && count > 0)
			{
				for (dvpUint32 i = 0; i < count; ++i)
				{
					dvpCameraInfo info = { 0 };
					if (dvpEnum(i, &info) == DVP_STATUS_OK)
					{
						std::string linkName = info.LinkName;
						std::string ip;
						auto pos = linkName.find('-');
						if (pos != std::string::npos && pos + 1 < linkName.size()) {
							ip = linkName.substr(pos + 1);
						}
						else {
							ip = linkName;
						}
						ipList.emplace_back(ip);
					}
				}
			}
			return ipList;
		}

		std::vector<CameraInfo> Camera_DS::getCameraInfoList()
		{
			std::vector<CameraInfo> cameraInfoList;
			dvpUint32 count = 0;
			if (dvpRefresh(&count) == DVP_STATUS_OK && count > 0)
			{
				for (dvpUint32 i = 0; i < count; ++i)
				{
					dvpCameraInfo info = { 0 };
					if (dvpEnum(i, &info) == DVP_STATUS_OK)
					{
						CameraInfo cameraInfo;
						// 只提取IP部分
						std::string linkName = info.LinkName;
						auto pos = linkName.find('-');
						if (pos != std::string::npos && pos + 1 < linkName.size()) {
							cameraInfo.ip = linkName.substr(pos + 1);
						}
						else {
							cameraInfo.ip = linkName;
						}
						cameraInfo.name = "DS"; // 设备友好名
						cameraInfo.mac = info.PortInfo;      // 端口信息，部分型号为MAC
						cameraInfoList.emplace_back(std::move(cameraInfo));
					}
				}
			}
			return cameraInfoList;
		}

		bool Camera_DS::connectCamera()
		{
			dvpUint32 count = 0;
			if (dvpRefresh(&count) != DVP_STATUS_OK || count == 0)
				return false;

			dvpCameraInfo info = { 0 };
			int foundIndex = -1;
			std::string foundIp;
			for (dvpUint32 i = 0; i < count; ++i)
			{
				if (dvpEnum(i, &info) == DVP_STATUS_OK)
				{
					// 提取info.LinkName中的IP部分
					std::string linkName = info.LinkName;
					std::string ip;
					auto pos = linkName.find('-');
					if (pos != std::string::npos && pos + 1 < linkName.size()) {
						ip = linkName.substr(pos + 1);
					}
					else {
						ip = linkName;
					}
					// 比较IP部分
					if (_ip == ip)
					{
						foundIndex = static_cast<int>(i);
						foundIp = ip;
						break;
					}
				}
			}
			if (foundIndex == -1)
				return false;

			dvpHandle handle = 0;
			if (dvpOpen(foundIndex, OPEN_NORMAL, &handle) != DVP_STATUS_OK)
				return false;

			m_cameraHandle = handle;
			_cameraInfo.ip = foundIp; 
			_cameraInfo.name = info.FriendlyName;
			_cameraInfo.mac = info.PortInfo;

			return true;
		}

		bool Camera_DS::disconnectCamera()
		{
			if (!m_cameraHandle) {
				return true;
			}

			if (_isMonitor) {
				dvpStop(m_cameraHandle);
				_isMonitor = false;
			}

			dvpStatus status = dvpClose(m_cameraHandle);
			if (status == DVP_STATUS_OK) {
				m_cameraHandle = 0;
				return true;
			}
			else {
				return false;
			}
		}

		bool Camera_DS::getConnectState(bool& isGet)
		{
			if (!m_cameraHandle) {
				isGet = false;
				return false;
			}

			dvpCameraInfo info = { 0 };
			dvpStatus status = dvpGetCameraInfo(m_cameraHandle, &info);
			if (status == DVP_STATUS_OK) {
				isGet = true;
				return true;
			}
			else {
				isGet = false;
				return false;
			}
		}

		bool Camera_DS::setFrameRate(float cameraFrameRate)
		{
			if (!m_cameraHandle) {
				return false;
			}

			// 推荐优先使用 dvpSetFrameRate，如果没有则用 dvpSetConfigString
#ifdef dvpSetFrameRate
			dvpStatus status = dvpSetFrameRate(m_cameraHandle, cameraFrameRate);
			if (status == DVP_STATUS_OK) {
				return true;
			}
#else
	// 通过属性名设置帧率
			dvpStatus status = dvpSetConfigString(m_cameraHandle, "AcquisitionFrameRate", std::to_string(cameraFrameRate).c_str());
			if (status == DVP_STATUS_OK) {
				return true;
			}
#endif

			return false;
		}

		float Camera_DS::getFrameRate(bool& isGet)
		{
			if (!m_cameraHandle) {
				isGet = false;
				return 0.0f;
			}

			// 推荐优先使用 dvpGetFrameRate，如果没有则用 dvpGetConfigString
#ifdef dvpGetFrameRate
			float frameRate = 0.0f;
			dvpStatus status = dvpGetFrameRate(m_cameraHandle, &frameRate);
			if (status == DVP_STATUS_OK) {
				isGet = true;
				return frameRate;
			}
#else
	// 通过属性名获取帧率
			const char* szValue = nullptr;
			dvpStatus status = dvpGetConfigString(m_cameraHandle, "AcquisitionFrameRate", &szValue);
			if (status == DVP_STATUS_OK && szValue) {
				isGet = true;
				return static_cast<float>(std::atof(szValue));
			}
#endif

			isGet = false;
			return 0.0f;
		}

		bool Camera_DS::startMonitor()
		{
			if (!m_cameraHandle) {
				return false;
			}
			if (_isMonitor) {
				return false;
			}
			dvpStatus status = dvpStart(m_cameraHandle);
			if (status == DVP_STATUS_OK) {
				_isMonitor = true;
				return true;
			}
			return false;
		}

		bool Camera_DS::stopMonitor()
		{
			if (!m_cameraHandle) {
				return false;
			}
			if (!_isMonitor) {
				return false;
			}
			dvpStatus status = dvpStop(m_cameraHandle);
			if (status == DVP_STATUS_OK) {
				_isMonitor = false;
				return true;
			}
			return false;
		}

		bool Camera_DS::setHeartbeatTime(size_t heartBeatTime)
		{
			/*if (!m_cameraHandle) {
				return false;
			}
		
			dvpStatus status = dvpSetConfigString(m_cameraHandle, "HeartbeatTimeout", std::to_string(heartBeatTime).c_str());
			return status == DVP_STATUS_OK;*/
			//该相机不支持
			return false;

		}

		size_t Camera_DS::getHeartbeatTime(bool& isGet)
		{
			/*if (!m_cameraHandle) {
				isGet = false;
				return 0;
			}

			const char* szValue = nullptr;
			dvpStatus status = dvpGetConfigString(m_cameraHandle, "HeartbeatTimeout", &szValue);
			if (status == DVP_STATUS_OK && szValue) {
				isGet = true;
				return static_cast<size_t>(std::strtoull(szValue, nullptr, 10));
			}
			else {
				isGet = false;
				return 0;
			}*/
			return false;
			//该相机不支持
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
			//对于线扫相机弃用
			return false;
		}

		bool Camera_DS::setInTriggerLine(size_t lineIndex)
		{
			return false;
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
			return CameraTriggerMode::HardwareTriggered;
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

		bool Camera_DS::setFrameTriggered(bool state)
		{
			if (!m_cameraHandle)
			{
				return false;

			}

			auto status = dvpSetTriggerState(m_cameraHandle, state);

			return status == DVP_STATUS_OK;
		}

		bool Camera_DS::getFrameTriggered(bool& isGet)
		{
			if (!m_cameraHandle)
			{
				isGet = false;
				return false;
			}

			bool state = false;
			dvpStatus status = dvpGetTriggerState(m_cameraHandle, &state);
			if (status == DVP_STATUS_OK)
			{
				isGet = true;
				return state;
			}
			else
			{
				isGet = false;
				return false;
			}
		}

		bool Camera_DS::setLineTriggered(bool state)
		{
			if (!m_cameraHandle)
			{
				return false;

			}

			std::string s = state ? "true" : "false";
			auto status = dvpSetConfigString(m_cameraHandle, "LineTrigEnable", s.c_str());

			return status == DVP_STATUS_OK;
		}

		bool Camera_DS::getLineTriggered(bool& isGet)
		{
			if (!m_cameraHandle)
			{
				isGet = false;
				return false;
			}

			const char* szValue = nullptr;
			dvpStatus status = dvpGetConfigString(m_cameraHandle, "LineTrigEnable", &szValue);
			if (status == DVP_STATUS_OK && szValue)
			{
				if (std::string(szValue) == "true")
				{
					isGet = true;
					return true;
				}
				else
				{
					isGet = true;
					return false;
				}
			}
			else
			{
				isGet = false;
				return false;
			}
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

		bool Camera_DS::setLineHeight(size_t number)
		{
			if (!m_cameraHandle) {
				return false; 
			}

			dvpStatus status = dvpSetConfigString(m_cameraHandle, "LineHeight", std::to_string(number).c_str());
			if (status == DVP_STATUS_OK) {
				return true; 
			}
			else {
				return false; 
			}
		}

		size_t Camera_DS::getLineHeight(bool& isGet)
		{
			if (!m_cameraHandle) {
				isGet = false; 
				return 0;
			}

			const char* szValue = nullptr;
			dvpStatus status = dvpGetConfigString(m_cameraHandle, "LineHeight", &szValue);
			if (status == DVP_STATUS_OK && szValue) {
				isGet = true;
				return static_cast<size_t>(std::strtoull(szValue, nullptr, 10));
			}
			else {
				isGet = false; 
				return 0;
			}
		}

		Camera_DS_Active::Camera_DS_Active()
		{

		}

		Camera_DS_Active::~Camera_DS_Active()
		{
		}

		cv::Mat Camera_DS_Active::getImage(bool& isGet)
		{

			return cv::Mat();
		}

		cv::Mat Camera_DS_Active::getImage()
		{

			return cv::Mat();
		}

		Camera_DS_Passive::Camera_DS_Passive(UserToCallBack userToCallback)
			: _userToCallBack(std::move(userToCallback))
		{
		}

		Camera_DS_Passive::~Camera_DS_Passive()
		{
			if (_isMonitor)
			{
				this->Camera_DS_Passive::stopMonitor();
			}
		}

		bool Camera_DS_Passive::RegisterCallBackFunc()
		{
			if (!m_cameraHandle) {
				return false;
			}

			dvpStatus status = dvpRegisterStreamCallback(
				m_cameraHandle,
				&Camera_DS_Passive::ImageCallBackFunc,
				STREAM_EVENT_FRAME_THREAD,
				this
			);
			return status == DVP_STATUS_OK;

		}

		dvpInt32 Camera_DS_Passive::ImageCallBackFunc(dvpHandle handle, dvpStreamEvent event, void* pContext,
			dvpFrame* pFrame, void* pBuffer)
		{
			auto* pThis = static_cast<Camera_DS_Passive*>(pContext);
			if (!pThis || !pFrame || !pBuffer) {
				return 0;
			}

			cv::Mat mat = ImageFrameConvert::DS_ConvertFrameToMat(*pFrame, pBuffer);

			if (!mat.empty() && pThis->_userToCallBack) {
				pThis->_userToCallBack(mat);
			}

			return 0;
		}

	}
}