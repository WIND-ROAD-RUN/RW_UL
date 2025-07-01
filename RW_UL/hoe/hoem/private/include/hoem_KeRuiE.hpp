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
		public:
			bool connect() override;
			bool disconnect() override;
			bool isConnected() const override;
			bool getIState(ModbusI locate) const override;
			bool getOState(ModbusO locate) const override;
			bool reconnect() override;
			bool setIState(ModbusI locate, bool state) override;
			bool setOState(ModbusO locate, bool state) override;
		};
	}
}