#pragma once

#include <string>
#include <vector>

#include"hoei_CPUInfo.hpp"
#include"hoei_MotherBoardInfo.hpp"

namespace rw {
	namespace hoei {
		struct HardwareInfo
		{
		public:
			CPUInfo cpu{};
			MotherBoardInfo motherBoard{};
		public:
			HardwareInfo() = default;
		};

	}
}
