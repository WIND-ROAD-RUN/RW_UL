#include "hoec_Camera_iRay.hpp"
#include"IMV/IMVApi.h"
#include"IMV/IMVDefines.h"
#include"IMVFG/IMVFGApi.h"
#include"IMVFG/IMVFGDefines.h"

namespace rw
{
	namespace hoec_v1
	{

		bool Camera_iRay::connectCamera()
		{



			auto ret = IMV_EnumDevices(&m_deviceInfoList, interfaceTypeAll);
			if (IMV_OK != ret)
			{
				printf("Enumeration devices failed! ErrorCode[%d]\n", ret);
				return false;
			}
			//没有找到相机
			if (m_deviceInfoList.nDevNum < 1)
			{
				return false;

			}
			else
			{

				for (unsigned int nIndex = 0; nIndex < m_deviceInfoList.nDevNum; nIndex++)
				{
					// 获取设备的 IP 地址
					const char* ipAddress = m_deviceInfoList.pDevInfo[nIndex].InterfaceInfo.gigeInterfaceInfo.ipAddress;
					if (_ip == std::string(ipAddress))
					{

						ret = IMV_CreateHandle(&m_devHandle, modeByCameraKey, (void*)nIndex);
						if (IMV_OK != ret)
						{
							printf("create devHandle failed! cameraKey[%s], ErrorCode[%d]\n", nIndex, ret);
							return false;
						}

						// 打开相机 
						// Open camera 
						ret = IMV_Open(m_devHandle);
						if (IMV_OK != ret)
						{
							printf("open camera failed! ErrorCode[%d]\n", ret);
							return false;
						}




						return true;
					}
				}
			}
		}
		bool Camera_iRay::getConnectState(bool& isGet)
		{

			return false;

		}

		bool Camera_iRay::setHeartbeatTime(size_t heartBeatTime)
		{
			int ret = IMV_OK;
			// 设置心跳超时时间，单位为毫秒
			ret = IMV_SetIntFeatureValue(m_devHandle, "GevHeartbeatTimeout", static_cast<int64_t>(heartBeatTime));
			if (IMV_OK != ret)
			{
				printf("set GevHeartbeatTimeout value = %zu fail, ErrorCode[%d]\n", heartBeatTime, ret);
				return false;
			}
			return true;
		}

		size_t Camera_iRay::getHeartbeatTime(bool& isGet)
		{
			int ret = IMV_OK;
			int64_t heartbeatTime = 0;
			// 获取心跳超时时间，单位为毫秒
			ret = IMV_GetIntFeatureValue(m_devHandle, "GevHeartbeatTimeout", &heartbeatTime);
			if (IMV_OK != ret)
			{
				printf("get GevHeartbeatTimeout fail, ErrorCode[%d]\n", ret);
				isGet = false;
				return 0;
			}
			isGet = true;
			return static_cast<size_t>(heartbeatTime);
		}
		bool Camera_iRay::setFrameRate(float cameraFrameRate)
		{
			if (!m_devHandle)
			{
				printf("Device handle is null!\n");
				return false;
			}

			int ret = IMV_SetDoubleFeatureValue(m_devHandle, "AcquisitionFrameRate", static_cast<double>(cameraFrameRate));
			if (IMV_OK != ret)
			{
				printf("set AcquisitionFrameRate value = %.2f fail, ErrorCode[%d]\n", cameraFrameRate, ret);
				return false;
			}

			return true;
		}
		float Camera_iRay::getFrameRate(bool& isGet)
		{
			int ret = IMV_OK;
			double frameRate = 0.0;
			// 获取帧率
			ret = IMV_GetDoubleFeatureValue(m_devHandle, "AcquisitionFrameRate", &frameRate);
			if (IMV_OK != ret)
			{
				printf("get AcquisitionFrameRate fail, ErrorCode[%d]\n", ret);
				isGet = false;
				return 0.0f;
			}
			isGet = true;
			return static_cast<float>(frameRate);
		}

		bool Camera_iRay::setExposureTime(size_t value)
		{
			int ret = IMV_OK;

			ret = IMV_SetDoubleFeatureValue(m_devHandle, "ExposureTime", value);
			if (IMV_OK != ret)
			{
				printf("set ExposureTime value = %0.2f fail, ErrorCode[%d]\n", value, ret);
				return false;
			}

			return true;
		}
		size_t Camera_iRay::getExposureTime(bool& isGet)
		{
			int ret = IMV_OK;
			double exposureTime = 0.0;
			// 获取曝光时间
			ret = IMV_GetDoubleFeatureValue(m_devHandle, "ExposureTime", &exposureTime);
			if (IMV_OK != ret)
			{
				printf("get ExposureTime fail, ErrorCode[%d]\n", ret);
				isGet = false;
				return 0;
			}
			isGet = true;
			return static_cast<size_t>(exposureTime);
		}


		bool Camera_iRay::setGain(size_t value)
		{
			int ret = IMV_OK;

			ret = IMV_SetDoubleFeatureValue(m_devHandle, "GainRaw", value);
			if (IMV_OK != ret)
			{
				printf("set GainRaw value = %0.2f fail, ErrorCode[%d]\n", value, ret);
				return false;
			}

			return true;


		}
		bool Camera_iRay::setInTriggerLine(size_t lineIndex)
		{
			if (!m_devHandle)
			{
				printf("Device handle is null!\n");
				return false;
			}

			// 只支持 Line1~Line4
			if (lineIndex < 1 || lineIndex > 4)
			{
				printf("Invalid line index: %zu\n", lineIndex);
				return false;
			}

			std::string lineName = "Line" + std::to_string(lineIndex);

			int ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerSource", lineName.c_str());
			if (IMV_OK != ret)
			{
				printf("Set TriggerSource to %s failed! ErrorCode[%d]\n", lineName.c_str(), ret);
				return false;
			}

			return true;
		}

		bool Camera_iRay::setTriggerMode(CameraTriggerMode mode)
		{
			if (!m_devHandle)
			{
				printf("Device handle is null!\n");
				return false;
			}

			int ret = IMV_OK;

			if (mode == CameraTriggerMode::SoftwareTriggered)
			{
				// 软触发
				ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerSource", "Software");
				if (IMV_OK != ret)
				{
					printf("Set TriggerSource to Software failed! ErrorCode[%d]\n", ret);
					return false;
				}
			}
			else if (mode == CameraTriggerMode::HardwareTriggered)
			{
				// 硬触发，默认用Line1
				ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerSource", "Line1");
				if (IMV_OK != ret)
				{
					printf("Set TriggerSource to Line1 failed! ErrorCode[%d]\n", ret);
					return false;
				}
			}
			else
			{
				printf("Unknown trigger mode!\n");
				return false;
			}

			// 设置触发选择器为 FrameStart
			ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerSelector", "FrameStart");
			if (IMV_OK != ret)
			{
				printf("Set TriggerSelector to FrameStart failed! ErrorCode[%d]\n", ret);
				return false;
			}

			// 设置触发模式为 On
			ret = IMV_SetEnumFeatureSymbol(m_devHandle, "TriggerMode", "On");
			if (IMV_OK != ret)
			{
				printf("Set TriggerMode to On failed! ErrorCode[%d]\n", ret);
				return false;
			}

			return true;
		}
		bool Camera_iRay::setFrameTriggered(bool state)
		{
			return false;
		}
		bool Camera_iRay::getFrameTriggered(bool& isGet)
		{
			return false;
		}
		size_t Camera_iRay::getGain(bool& isGet)
		{
			int ret = IMV_OK;
			double gainValue = 0.0;
			// 获取增益值
			ret = IMV_GetDoubleFeatureValue(m_devHandle, "GainRaw", &gainValue);
			if (IMV_OK != ret)
			{
				printf("get GainRaw fail, ErrorCode[%d]\n", ret);
				isGet = false;
				return 0;
			}
			isGet = true;
			return static_cast<size_t>(gainValue);
		}
		bool Camera_iRay::setLineTriggered(bool state)
		{
			return false;
		}

		bool Camera_iRay::getLineTriggered(bool& isGet)
		{
			return false;
		}

		bool Camera_iRay::setPreDivider(size_t number)
		{
			return false;
		}

		bool Camera_iRay::setMultiplier(size_t number)
		{
			return false;
		}

		bool Camera_iRay::setPostDivider(size_t number)
		{
			return false;
		}
		bool Camera_iRay::getEncoderNumber(double& number)
		{
			return false;
		}
		bool Camera_iRay::setLineHeight(size_t number)
		{
			return false;
		}
		size_t Camera_iRay::getLineHeight(bool& isGet)
		{
			isGet = false;
			return size_t();
		}

		
		
	}
}