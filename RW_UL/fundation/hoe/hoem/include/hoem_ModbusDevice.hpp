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
			//使用前请线了解了对应plc的modbus地址映射规则，以及物理寄存器的位数
			//本类默认从站ID为1，如有需要请自行扩展
			//Please understand the Modbus address mapping rules of the corresponding PLC and the number of physical registers before using it.
			//This class defaults to slave ID 1, please extend it yourself if needed.
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

			bool writeRegister(Address16 startAddress, float value, Endianness byteOrder);
			bool writeRegisters(Address16 startAddress, const std::vector<float>& data, Endianness byteOrder);
			bool readRegister(Address16 startAddress, float& value, Endianness byteOrder);
			bool readRegisters(Address16 startAddress, std::vector<float>& values, Endianness byteOrder);
		};
	}
}