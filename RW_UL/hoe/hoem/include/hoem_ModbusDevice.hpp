#pragma once

#include"hoem_utilty.hpp"

typedef struct _modbus modbus_t;

namespace rw
{
	namespace hoem
	{
		class ModbusDevice{
		private:
			modbus_t* _modbusContext = nullptr;
			std::string _ip;
			int _port = 0;
			Address _baseAddress = 0;
		public:
			ModbusDevice(const std::string& ip, int port, Address baseAddress = 0);
			~ModbusDevice();
		public:
			bool connect() ;
			void disconnect() ;
			bool isConnected() const ;
			bool readRegisters(Address startAddress, Quantity quantity, std::vector<RegisterValue>& data) ;
			bool writeRegisters(Address startAddress, const std::vector<RegisterValue>& data) ;
			bool readCoils(Address startAddress, Quantity quantity, std::vector<bool>& data) ;
			bool writeCoil(Address address, bool state) ;
			bool writeCoils(Address startAddress, const std::vector<bool>& states) ;
			bool setBasedAddress(Address basedAddress) ;
			Address getBasedAddress() const ;
			bool readRegistersAbsolute(Address address, Quantity quantity, std::vector<RegisterValue>& data) ;
			bool writeRegistersAbsolute(Address address, const std::vector<RegisterValue>& data) ;
		};
	}
}
