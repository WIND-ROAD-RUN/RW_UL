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

		enum class CameraTriggerMode
		{
			SoftwareTriggered,
			HardwareTriggered,
		};

		enum class CameraProvider
		{
			MVS
		};
	}
}