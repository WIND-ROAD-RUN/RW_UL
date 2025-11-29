#pragma once

#include"hoem_utilty.hpp"

typedef struct _modbus modbus_t;

namespace rw
{
	namespace hoem
	{
		struct ModbusDeviceTcpCfg
		{
			std::string ip;
			int port;
			Address16 baseAddress = 0;
		};

		struct ModbusDeviceRtuCfg
		{
			std::string device;
			int baud;
			char parity;
			int dataBit;
			int stopBit;
			Address16 baseAddress = 0;
		};

		class ModbusDevice {
		private:
			modbus_t* _modbusContext = nullptr;
			Address16 _baseAddress = 0;
		public:
			ModbusDevice(const std::string& ip, int port, Address16 baseAddress = 0);
			ModbusDevice(const ModbusDeviceTcpCfg & cfg);
			ModbusDevice(const ModbusDeviceRtuCfg& cfg);
			~ModbusDevice();
		public:
			bool connect();
			bool disconnect();
			bool isConnected() const;
			bool reconnect();
			bool readRegisters(Address16 startAddress, Quantity quantity, std::vector<UInt16>& data);
			bool readRegister(Address16 startAddress, UInt32& data, Endianness byteOrder);
			bool readRegisters(Address16 startAddress, std::vector<UInt32>& data, Endianness byteOrder);

			bool writeRegisters(Address16 startAddress, const std::vector<UInt16>& data);
			bool writeRegister(Address16 startAddress, UInt32 data, Endianness byteOrder);
			bool writeRegisters(Address16 startAddress, const std::vector<UInt32>& data, Endianness byteOrder);

			bool readCoils(Address16 startAddress, Quantity quantity, std::vector<bool>& data);
			bool writeCoil(Address16 address, bool state);
			bool writeCoils(Address16 startAddress, const std::vector<bool>& states);
			bool setBasedAddress(Address16 basedAddress);
			Address16 getBasedAddress() const;
			bool readRegistersAbsolute(Address16 address, Quantity quantity, std::vector<UInt16>& data);
			bool writeRegistersAbsolute(Address16 address, const std::vector<UInt16>& data);
		};
	}
}