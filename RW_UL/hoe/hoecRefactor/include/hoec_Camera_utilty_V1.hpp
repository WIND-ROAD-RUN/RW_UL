#pragma once

#include <string>
#include<functional>

namespace cv {
	class Mat;
}

namespace rw
{
	namespace hoec_v1
	{
		struct OutTriggerConfig
		{
		public:
			size_t lineSelector{ 0 };
			size_t lineMode{ 0 };
			size_t lineSource{ 0 };
			long durationValue{ 0 };
			size_t delayValue{ 0 };
			size_t preDelayValue{ 0 };
			bool strobeEnable = false;
		};

		struct CameraInfo
		{
			std::string ip;
			std::string name;
			std::string mac;
		};

		enum class CameraTakePictureMode
		{
			Active,
			Passive
		};

		inline const char* to_string(CameraTakePictureMode e)
		{
			switch (e)
			{
			case CameraTakePictureMode::Active: return "Active";
			case CameraTakePictureMode::Passive: return "Passive";
			default: return "unknown";
			}
		}

		enum class CameraTriggerMode
		{
			SoftwareTriggered,
			HardwareTriggered,
		};

		inline const char* to_string(CameraTriggerMode e)
		{
			switch (e)
			{
			case CameraTriggerMode::SoftwareTriggered: return "SoftwareTriggered";
			case CameraTriggerMode::HardwareTriggered: return "HardwareTriggered";
			default: return "unknown";
			}
		}

		enum class CameraProvider
		{
			MVS,
			DS
		};

		inline CameraProvider from_string(const std::string &s)
		{
			if (s=="MVS")
			{
				return CameraProvider::MVS;
			}
			else
			{
				return CameraProvider::DS;
			}
		}

		inline const char* to_string(CameraProvider e)
		{
			switch (e)
			{
			case CameraProvider::MVS: return "MVS";
			case CameraProvider::DS: return "DS";
			default: return "unknown";
			}
		}
	}
}