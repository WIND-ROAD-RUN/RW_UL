#pragma once

#include<memory>

#include"rqw_CameraObjectCore.hpp"

#include<opencv2/opencv.hpp>

#include<QVector>
#include<QObject>
#include<memory>

#include"scc_motion.h"

namespace zwy {
	namespace scc {
		class GlobalMotion
			:public QObject
		{
			Q_OBJECT
		public:
			std::unique_ptr<Motion> motionPtr;
		public:
			static GlobalMotion& getInstance()
			{
				static GlobalMotion instance;
				return instance;
			}

			GlobalMotion(const GlobalMotion&) = delete;
			GlobalMotion& operator=(const GlobalMotion&) = delete;

		private:
			GlobalMotion();
		public:
			~GlobalMotion() = default;
		};
	}
}

namespace rw {
	namespace hoec
	{
		class CameraPassive;
	}

	namespace rqw {
		class CameraPassiveObject : public QObject
		{
			Q_OBJECT
		public:
			size_t motionRedix;
			size_t motionInde;
		private:
			std::unique_ptr<hoec::CameraPassive> _cameraPassive;
			CameraMetaData _cameraMetaData;
		public:
			CameraPassiveObject(const CameraPassiveObject&) = delete;
			CameraPassiveObject& operator=(const CameraPassiveObject&) = delete;
			CameraPassiveObject(CameraPassiveObject&&) = delete;
			CameraPassiveObject& operator=(CameraPassiveObject&&) = delete;
		public:
			CameraPassiveObject(QObject* parent = nullptr);
			~CameraPassiveObject() override;
		public:
			void startMonitor() const;
			void stopMonitor() const;
		public:
			//TODO: Add more functions
			void setHeartbeatTime(size_t value) const;
			void setFrameRate(float value) const;
			size_t getHeartbeatTime() const;
			float getFrameRate() const;
			void setExposureTime(size_t value) const;
			void setGain(size_t value) const;
			void setIOTime(size_t value) const;
			void setTriggerMode(CameraObjectTrigger mode) const;
			void setTriggerLine(size_t lineIndex)const;
		public:
			//TODO: Add more functions
			[[nodiscard]] size_t getExposureTime() const;
			[[nodiscard]] size_t getGain() const;
			[[nodiscard]] size_t getIOTime() const;
			[[nodiscard]] CameraObjectTrigger getMonitorMode() const;
			[[nodiscard]] size_t getTriggerLine() const;
		public:
			void initCamera(const CameraMetaData& cameraMetaData, CameraObjectTrigger triggerMode);
			bool getConnectState();
		signals:
			void frameCaptured(cv::Mat frame, float);
		};
	} // namespace rqw
} // namespace rw