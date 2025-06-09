#pragma once

#include"opencv2/opencv.hpp"

#include"hoec_Camera_v1.hpp"

#include<vector>
#include<string>
#include<functional>
#include<atomic>


namespace rw
{
	namespace hoec_v1
	{
		class Camera_DS :
			public ICamera {
		private:
			static std::atomic<size_t> _cameraNum;
		public:
			Camera_DS();
			~Camera_DS() override;
		public:
			static bool _isIniSDK;
			static std::vector<std::string> getCameraIpList();
			static std::vector<CameraInfo> getCameraInfoList();
			static bool initSDK();
			static bool unInitSDK();
		public:
			bool connectCamera() override;
			bool getConnectState(bool& isGet)override;
			bool setFrameRate(float cameraFrameRate) override;
			float getFrameRate(bool& isGet) override;
		public:
			bool startMonitor() override;
			bool stopMonitor() override;
		public:
			bool setHeartbeatTime(size_t heartBeatTime)override;
			size_t getHeartbeatTime(bool& isGet) override;
			bool setExposureTime(size_t value) override;
			bool setGain(size_t value) override;
			bool setTriggerMode(CameraTriggerMode mode) override;
			bool setInTriggerLine(size_t lineIndex) override;
		public:
			size_t getExposureTime(bool& isGet) override;
			size_t getGain(bool& isGet) override;
			CameraTriggerMode getMonitorMode(bool& isGet) override;
			size_t getTriggerLine(bool& isGet) override;
		public:
			bool setOutTriggerConfig(const OutTriggerConfig& config) override;
			bool outTrigger() override;
			bool outTrigger(bool isOpen) override;
		protected:
			void* m_cameraHandle{ nullptr };
		protected:
			bool _isMonitor{ false };
			CameraTriggerMode triggerMode;
		};

		class Camera_DS_Active
			:public Camera_DS, public ICameraActive {
		public:
			Camera_DS_Active();
			~Camera_DS_Active() override;
		public:
			cv::Mat getImage(bool& isGet) override;
			cv::Mat getImage() override;
		};

		class Camera_DS_Passive
			:public Camera_DS, public ICameraPassive {
		private:
			UserToCallBack _userToCallBack;
		public:
			Camera_DS_Passive(UserToCallBack userToCallback = [](cv::Mat mat) {
				std::cout << "No callback function" << std::endl;
				return;
				});
			~Camera_DS_Passive() override;
		public:
			bool RegisterCallBackFunc() override;
		public:

			static void __stdcall ImageCallBackFunc(/*unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser*/);
		};
	}
}