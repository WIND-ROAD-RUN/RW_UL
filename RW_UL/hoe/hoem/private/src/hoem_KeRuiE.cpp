#include"hoem_KeRuiE.hpp"

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
			/*if (_modbusDevice) {
				std::vector<bool> data;
				if (_modbusDevice->readCoils(locate.address, 1, data)) {
					return data[0];
				}
			}
			return false;*/
			return true;
		}

		bool KeRuiE::getOState(ModbusO locate) const
		{
			std::vector<RegisterValue> value(2);
			_modbusDevice->readRegisters(0, 2, value);

			auto int32=fromRegisterValuesToInt32(value,Endianness::BigEndian);
			auto int32O=modbusOToInt32(locate, Endianness::BigEndian);
			bool result = (int32 & int32O);

			return result;
		}

		bool KeRuiE::reconnect()
		{
			if (_modbusDevice) {
				return _modbusDevice->reconnect();
			}
			return false;
		}

		bool KeRuiE::setIState(ModbusI locate, bool state)
		{
			return 0;
		}

		bool KeRuiE::setOState(ModbusO locate, bool state)
		{
			return 0;
		}
	}
}