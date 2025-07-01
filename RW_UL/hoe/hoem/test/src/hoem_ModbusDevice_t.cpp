#include"hoem_pch_t.hpp"

#include"hoem_ModbusDevice.hpp"

namespace hoem_ModbusDevice
{
	TEST(ModbusDeviceTest,a)
	{
		rw::hoem::ModbusDevice device("192.168.1.199", 502, 0x20);
		auto connectResult=device.connect();

	}
}
