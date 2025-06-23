#pragma once

#include <string>

namespace rw {
	namespace hoei {
		struct MotherBoardInfo
		{
		public:
			MotherBoardInfo();
		public:
			std::string UUID{};
		public:
			static std::string GetMotherboardUniqueID();
		};


	}
}
