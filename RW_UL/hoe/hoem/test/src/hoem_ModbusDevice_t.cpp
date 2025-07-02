#include "hoem_KeRuiE.hpp"
#include"hoem_pch_t.hpp"

#include"hoem_ModbusDevice.hpp"

namespace hoem_ModbusDevice
{
	TEST(ModbusDeviceTest,a)
	{
		rw::hoem::ModbusDevice device("192.168.1.199", 502, 0x20);
		auto connectResult=device.connect();

		if (!connectResult)
		{
			std::cout << "Please check the connection parameters." << std::endl;
			SUCCEED();
		}

		//device.writeRegisters(0, { {0x0004, 0x0000} });

		rw::hoem::KeRuiE keRui(&device);
		auto result =  keRui.getOState(rw::hoem::ModbusO::Y03);
	}
}
