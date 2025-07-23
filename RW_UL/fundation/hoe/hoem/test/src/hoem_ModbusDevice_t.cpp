#include "hoem_KeRuiE.hpp"
#include"hoem_pch_t.hpp"

#include"hoem_ModbusDevice.hpp"
#include"hoem_ModbusDeviceFactory.hpp"

namespace hoem_ModbusDevice
{
	TEST(ModbusDeviceTest, a)
	{
		rw::hoem::ModbusConfig config;
		config.ip = "192.168.1.199";
		config.port = 502;
		auto deviceKeRuiE = rw::hoem::ModbusDeviceFactory::createDevice(rw::hoem::ModbusDeviceName::keRuiE, config);

		auto connectResult = deviceKeRuiE->connect();
		deviceKeRuiE->setOState(rw::hoem::ModbusO::Y01, true);
		deviceKeRuiE->setOState(rw::hoem::ModbusO::Y02, false);
		deviceKeRuiE->setOState(rw::hoem::ModbusO::Y03, false);
		deviceKeRuiE->setOState(rw::hoem::ModbusO::Y04, true);
		deviceKeRuiE->setOState(rw::hoem::ModbusO::Y05, false);
		deviceKeRuiE->setOState(rw::hoem::ModbusO::Y06, false);

		auto o1 = deviceKeRuiE->getOState(rw::hoem::ModbusO::Y01);
		auto o2 = deviceKeRuiE->getOState(rw::hoem::ModbusO::Y02);
		auto o3 = deviceKeRuiE->getOState(rw::hoem::ModbusO::Y03);
		auto o4 = deviceKeRuiE->getOState(rw::hoem::ModbusO::Y04);
		auto o5 = deviceKeRuiE->getOState(rw::hoem::ModbusO::Y05);
		auto o6 = deviceKeRuiE->getOState(rw::hoem::ModbusO::Y06);

		while (true)
		{
			while (true)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 每次循环延迟 1 秒
				std::cout << "Y01: " << deviceKeRuiE->getIState(rw::hoem::ModbusI::X00) << std::endl;
			}
		}
	}
}