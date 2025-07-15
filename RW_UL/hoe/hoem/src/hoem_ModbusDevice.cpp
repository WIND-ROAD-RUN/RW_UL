#include"hoem_ModbusDevice.hpp"

#include <iostream>
#include <modbus.h>
#include <ostream>

namespace rw
{
	namespace hoem
	{
		ModbusDevice::ModbusDevice(const std::string& ip, int port, Address baseAddress)
			:_ip(ip), _port(port), _baseAddress(baseAddress)
		{
			_modbusContext = modbus_new_tcp(_ip.c_str(), port);
			if (_modbusContext == nullptr) {
				throw std::runtime_error("Failed to create Modbus context");
			}
			modbus_set_slave(_modbusContext, 1); // 设置从站ID为1
		}

		ModbusDevice::~ModbusDevice()
		{
			ModbusDevice::disconnect();
			if (_modbusContext != nullptr) {
				modbus_free(_modbusContext);
				_modbusContext = nullptr;
			}
		}

		bool ModbusDevice::connect()
		{
			if (modbus_connect(_modbusContext) == -1)
			{
				return false;
			}
			return true;
		}

		bool ModbusDevice::disconnect()
		{
			if (_modbusContext != nullptr) {
				modbus_close(_modbusContext);
				return true;
			}
			return false;
		}

		bool ModbusDevice::isConnected() const
		{
			if (_modbusContext == nullptr) {
				return false;
			}
			// 尝试读取一个寄存器，判断连接是否正常
			uint16_t temp = 0;
			int result = modbus_read_registers(_modbusContext, _baseAddress, 1, &temp);
			return result == 1;
		}

		bool ModbusDevice::reconnect()
		{
			if (disconnect()) {
				return connect();
			}
			return false;
		}

		bool ModbusDevice::readRegisters(Address startAddress, Quantity quantity, std::vector<RegisterValue>& data)
		{
			if (!isConnected()) {
				return false;
			}
			data.resize(quantity);
			int result = modbus_read_registers(_modbusContext, startAddress + _baseAddress, quantity, reinterpret_cast<uint16_t*>(data.data()));
			uint16_t number = static_cast<uint16_t>(data[0]);

			std::cout << "number" << number << std::endl;
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeRegisters(Address startAddress, const std::vector<RegisterValue>& data)
		{
			if (!isConnected()) {
				return false;
			}
			int result = modbus_write_registers(_modbusContext, startAddress + _baseAddress, data.size(), reinterpret_cast<const uint16_t*>(data.data()));
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::readCoils(Address startAddress, Quantity quantity, std::vector<bool>& data)
		{
			if (!isConnected()) {
				return false;
			}
			data.resize(quantity);
			uint8_t* coilData = new uint8_t[(quantity + 7) / 8];
			int result = modbus_read_bits(_modbusContext, startAddress + _baseAddress, quantity, coilData);
			if (result < 0) {
				delete[] coilData;
				return false;
			}
			for (size_t i = 0; i < quantity; ++i) {
				data[i] = (coilData[i / 8] >> (i % 8)) & 0x01;
			}
			delete[] coilData;
			return true;
		}

		bool ModbusDevice::writeCoil(Address address, bool state)
		{
			if (!isConnected()) {
				return false;
			}
			int result = modbus_write_bit(_modbusContext, address + _baseAddress, state ? 1 : 0);
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeCoils(Address startAddress, const std::vector<bool>& states)
		{
			if (!isConnected()) {
				return false;
			}
			size_t numBytes = (states.size() + 7) / 8;
			uint8_t* coilData = new uint8_t[numBytes]();
			for (size_t i = 0; i < states.size(); ++i) {
				if (states[i]) {
					coilData[i / 8] |= (1 << (i % 8));
				}
			}
			int result = modbus_write_bits(_modbusContext, startAddress + _baseAddress, states.size(), coilData);
			delete[] coilData;
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::setBasedAddress(Address basedAddress)
		{
			_baseAddress = basedAddress;
			return true;
		}

		Address ModbusDevice::getBasedAddress() const
		{
			return _baseAddress;
		}

		bool ModbusDevice::readRegistersAbsolute(Address address, Quantity quantity, std::vector<RegisterValue>& data)
		{
			if (!isConnected()) {
				return false;
			}
			data.resize(quantity);
			int result = modbus_read_registers(_modbusContext, address, quantity, reinterpret_cast<uint16_t*>(data.data()));
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeRegistersAbsolute(Address address, const std::vector<RegisterValue>& data)
		{
			if (!isConnected()) {
				return false;
			}
			int result = modbus_write_registers(_modbusContext, address, data.size(), reinterpret_cast<const uint16_t*>(data.data()));
			if (result < 0) {
				return false;
			}
			return true;
		}
	}
}
