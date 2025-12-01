
#include "hoem_ModbusDevice.hpp"
#include"hoem_pch_t.hpp"

#include"hoem_ModbusDeviceScheduler.hpp"


namespace hoem_ModbusDevice
{
	TEST(ModbusDeviceTest, a)
	{
		rw::hoem::ModbusDeviceTcpCfg cfg;
		cfg.ip = "192.168.10.2";
		cfg.port = 502;

		auto sharedPtr = std::make_shared<rw::hoem::ModbusDevice>(cfg);

		auto connectResult= sharedPtr->connect();
		EXPECT_EQ(connectResult, true);

		rw::hoem::ModbusDeviceScheduler scheduler(sharedPtr);
		auto writeResult= scheduler.writeRegisterFloatAsync(6000, 0.3598f, rw::hoem::Endianness::LittleEndian);

		writeResult.wait();

	}
}