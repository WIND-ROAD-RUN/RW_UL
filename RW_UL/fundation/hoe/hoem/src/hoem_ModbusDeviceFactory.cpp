#include "hoem_ModbusDeviceFactory.hpp"

#include"hoem_KeRuiE.hpp"

namespace rw
{
	namespace hoem
	{
		static std::unique_ptr<IModbusDevice> createKeRuiE(const ModbusDeviceName& type, const ModbusConfig& config);

		std::unique_ptr<IModbusDevice> ModbusDeviceFactory::createDevice(const ModbusDeviceName& type, const ModbusConfig& config)
		{
			switch (type)
			{
			case ModbusDeviceName::keRuiE:
				return createKeRuiE(type, config);
			default:
				throw std::runtime_error("Unsupported Modbus device type");
			}
		}

		std::unique_ptr<IModbusDevice> createKeRuiE(const ModbusDeviceName& type, const ModbusConfig& config)
		{
			auto device = new rw::hoem::ModbusDevice(config.ip, config.port, 0);
			return std::make_unique<KeRuiE>(device);
		}
	}
}