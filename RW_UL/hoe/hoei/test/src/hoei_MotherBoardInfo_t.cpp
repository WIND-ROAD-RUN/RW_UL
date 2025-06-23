#include"hoei_utilty_t.hpp"

#include"hoei_MotherBoardInfo.hpp"

namespace hoei_HardwareInfo
{
	TEST(MotherBoardInfo, getMotherBoardInfo)
	{
		std::cout << "MotherboardUniqueID:" << rw::hoei::MotherBoardInfo::GetMotherboardUniqueID() << std::endl;
	}
}
