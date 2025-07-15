#include "hoem_ModbusDeviceFactory.hpp"

#include"hoem_KeRuiE.hpp"

namespace rw
{
	namespace hoem
	{
		static std::unique_ptr<IModbusDevice> createKeRuiE(const ModbusDeviceName& type, const ModbusIConfig& config);

		std::unique_ptr<IModbusDevice> ModbusDeviceFactory::createModelEngine(const ModbusDeviceName& type, const ModbusIConfig& config)
		{
			switch (type)
			{
			case ModbusDeviceName::keRuiE:
				return createKeRuiE(type,config);
			default:
				throw std::runtime_error("Unsupported Modbus device type");
			}
		}

		std::unique_ptr<IModbusDevice> createKeRuiE(const ModbusDeviceName& type, const ModbusIConfig& config)
		{
			auto device = new rw::hoem::ModbusDevice(config.ip, config.port, 0);
			return std::make_unique<KeRuiE>(device);
		}
	}
}
