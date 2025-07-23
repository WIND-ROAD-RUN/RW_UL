#include"hoem_KeRuiE.hpp"

#include<cmath>

namespace rw {
	namespace hoem {
		Address KeRuiE::switchAddress(ModbusI locate)
		{
			return 0;
		}

		Address KeRuiE::switchAddress(ModbusO locate)
		{
			return 0;
		}

		KeRuiE::KeRuiE(ModbusDevice* modbusDevice)
			: _modbusDevice(modbusDevice) {
			for (int i = 0; i < 16; i++)
			{
				_ioOutState.push_back(false);
				_ioInState.push_back(false);
			}
		}

		KeRuiE::~KeRuiE()
		{
			delete _modbusDevice;
		}

		bool KeRuiE::connect()
		{
			if (_modbusDevice) {
				return _modbusDevice->connect();
			}
			return false;
		}

		bool KeRuiE::disconnect()
		{
			if (_modbusDevice) {
				return _modbusDevice->disconnect();
			}
			return false;
		}

		bool KeRuiE::isConnected() const
		{
			if (_modbusDevice) {
				return _modbusDevice->isConnected();
			}
			return false;
		}

		bool KeRuiE::getIState(ModbusI locate) const
		{
			if (!_modbusDevice) {
				return false;
			}

			int index = static_cast<int>(locate);
			if (index < 0 || index >= _ioInState.size()) {
				return false;
			}

			std::vector<RegisterValue> data(1);
			if (!_modbusDevice->readRegisters(0x10, 2, data)) {
				return false;
			}

			uint16_t number = static_cast<uint16_t>(data[0]);

			bool isTriggered = (number & (1 << index)) != 0;

			return isTriggered;
		}

		bool KeRuiE::getOState(ModbusO locate) const
		{
			if (!_modbusDevice) {
				return false;
			}

			int index = static_cast<int>(locate);
			if (index < 0 || index >= _ioOutState.size()) {
				return false;
			}

			uint16_t number = 0;
			std::vector<RegisterValue> data(1);
			if (!_modbusDevice->readRegisters(0x20, 1, data)) {
				return false;
			}

			number = static_cast<uint16_t>(data[0]);

			bool result = (number & (1 << index)) != 0;
			return result;
		}

		bool KeRuiE::reconnect()
		{
			if (_modbusDevice) {
				return _modbusDevice->reconnect();
			}
			return false;
		}

		bool KeRuiE::setOState(ModbusO locate, bool state)
		{
			if (!_modbusDevice) {
				return false;
			}

			int index = static_cast<int>(locate);
			if (index < 0 || index >= _ioInState.size()) {
				return false;
			}
			_ioInState[index] = state;

			uint16_t number = 0;
			for (int i = 0; i < _ioInState.size(); i++) {
				if (_ioInState[i]) {
					number += static_cast<uint16_t>((std::pow)(2, i));
				}
			}

			bool result = _modbusDevice->writeRegisters(0x20, { {number, 0x0000} });
			return result;
		}
	}
}