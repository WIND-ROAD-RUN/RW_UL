#pragma once
#include <memory>
#include"hoem_IModbusDevices.hpp"
#include"hoem_utilty.hpp"

namespace rw {
	namespace hoem {
		class ModbusDeviceFactory
		{
		public:
			static std::unique_ptr<IModbusDevice> createDevice(const ModbusDeviceName & type, const ModbusConfig& config);
		};
	}
}