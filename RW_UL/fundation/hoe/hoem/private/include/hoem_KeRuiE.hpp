#pragma once

#include"hoem_IModbusDevices.hpp"

#include"hoem_ModbusDevice.hpp"

namespace rw {
	namespace hoem {
		class KeRuiE : public IModbusDevice {
		public:
			Address switchAddress(ModbusI locate);
			Address switchAddress(ModbusO locate);
		private:
			ModbusDevice * _modbusDevice;
		public:
			KeRuiE(ModbusDevice* modbusDevice);
			~KeRuiE() override;

		private:
			std::vector<bool>_ioOutState;
			std::vector<bool>_ioInState;

		public:
			bool connect() override;
			bool disconnect() override;
			bool isConnected() const override;
			bool getIState(ModbusI locate) const override;
			bool getOState(ModbusO locate) const override;
			bool reconnect() override;
			bool setOState(ModbusO locate, bool state) override;
		};
	}
}