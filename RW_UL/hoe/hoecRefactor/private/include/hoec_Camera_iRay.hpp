#include"opencv2/opencv.hpp"

#include"hoec_Camera_v1.hpp"

#include<vector>
#include<string>
#include<functional>
#include<atomic>

#include"DVPCamera.h"
#include"dvpir.h"
#include"dvpParam.h"
#include "IMV/IMVDefines.h"

namespace rw
{
	namespace hoec_v1
	{
		class Camera_iRay :
			public ICamera {
		public:
			Camera_iRay();
			~Camera_iRay() override;

			// 相机句柄 | camera handle
			IMV_HANDLE	m_devHandle;
			// 发现的相机列表 | List of cameras found
			IMV_DeviceList m_deviceInfoList;	

		public:
			bool connectCamera() override;
			bool getConnectState(bool& isGet) override;
			bool disconnectCamera() override;
			bool startMonitor() override;
			bool stopMonitor() override;
			bool setHeartbeatTime(size_t heartBeatTime) override;
			size_t getHeartbeatTime(bool& isGet) override;
			bool setFrameRate(float cameraFrameRate) override;
			float getFrameRate(bool& isGet) override;
			bool setExposureTime(size_t value) override;
			bool setGain(size_t value) override;
			bool setInTriggerLine(size_t lineIndex) override;
			bool setTriggerMode(CameraTriggerMode mode) override;
			bool setFrameTriggered(bool state) override;
			bool getFrameTriggered(bool& isGet) override;
			bool setLineTriggered(bool state) override;
			bool getLineTriggered(bool& isGet) override;
			bool setPreDivider(size_t number) override;
			bool setMultiplier(size_t number) override;
			bool setPostDivider(size_t number) override;
			bool getEncoderNumber(double& number) override;
			bool setLineHeight(size_t number) override;
			size_t getLineHeight(bool& isGet) override;
			[[nodiscard]] size_t getExposureTime(bool& isGet) override;
			[[nodiscard]] size_t getGain(bool& isGet) override;
			[[nodiscard]] CameraTriggerMode getMonitorMode(bool& isGet) override;
			[[nodiscard]] size_t getTriggerLine(bool& isGet) override;
			bool setOutTriggerConfig(const OutTriggerConfig& config) override;
			bool outTrigger() override;
			bool outTrigger(bool isOpen) override;
		};
	}
}
