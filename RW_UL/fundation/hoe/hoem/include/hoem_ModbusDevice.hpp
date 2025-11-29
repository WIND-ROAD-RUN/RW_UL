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
			Address baseAddress = 0;
		};

		struct ModbusDeviceRtuCfg
		{
			std::string device;
			int baud;
			char parity;
			int dataBit;
			int stopBit;
			Address baseAddress = 0;
		};

		class ModbusDevice {
		private:
			modbus_t* _modbusContext = nullptr;
			Address _baseAddress = 0;
		public:
			ModbusDevice(const std::string& ip, int port, Address baseAddress = 0);
			ModbusDevice(const ModbusDeviceTcpCfg & cfg);
			ModbusDevice(const ModbusDeviceRtuCfg& cfg);
			~ModbusDevice();
		public:
			bool connect();
			bool disconnect();
			bool isConnected() const;
			bool reconnect();
			bool readRegisters(Address startAddress, Quantity quantity, std::vector<RegisterValue>& data);
			bool readRegister(Address startAddress, RegisterValue32& data, Endianness byteOrder);
			bool readRegisters(Address startAddress, std::vector<RegisterValue32>& data, Endianness byteOrder);

			bool writeRegisters(Address startAddress, const std::vector<RegisterValue>& data);
			bool writeRegister(Address startAddress, RegisterValue32 data, Endianness byteOrder);
			bool writeRegisters(Address startAddress, const std::vector<RegisterValue32>& data, Endianness byteOrder);

			bool readCoils(Address startAddress, Quantity quantity, std::vector<bool>& data);
			bool writeCoil(Address address, bool state);
			bool writeCoils(Address startAddress, const std::vector<bool>& states);
			bool setBasedAddress(Address basedAddress);
			Address getBasedAddress() const;
			bool readRegistersAbsolute(Address address, Quantity quantity, std::vector<RegisterValue>& data);
			bool writeRegistersAbsolute(Address address, const std::vector<RegisterValue>& data);
		};
	}
}