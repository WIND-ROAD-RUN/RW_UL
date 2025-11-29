
#include "hoem_ModbusDevice.hpp"
#include"hoem_pch_t.hpp"


namespace hoem_ModbusDevice
{
	TEST(ModbusDeviceTest, a)
	{
		rw::hoem::ModbusDeviceTcpCfg cfg;
		cfg.ip = "192.168.10.2";
		cfg.port = 502;
		rw::hoem::ModbusDevice device(cfg);
		auto connectResult=device.connect();
		EXPECT_EQ(connectResult, true);
	}
}