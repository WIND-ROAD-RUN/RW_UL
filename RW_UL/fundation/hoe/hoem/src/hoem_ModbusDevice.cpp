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

		ModbusDevice::ModbusDevice(const ModbusDeviceTcpCfg& cfg)
		{
			_ip = cfg.ip;
			_modbusContext = modbus_new_tcp(cfg.ip.c_str(), cfg.port);
			if (_modbusContext == nullptr) {
				throw std::runtime_error("Failed to create Modbus context");
			}
			modbus_set_slave(_modbusContext, 1); // 设置从站ID为1
		}

		ModbusDevice::ModbusDevice(const ModbusDeviceRtuCfg& cfg)
		{
			_modbusContext = modbus_new_rtu(cfg.device.c_str(), cfg.baud, cfg.parity, cfg.dataBit, cfg.stopBit);
			if (_modbusContext == nullptr) {
				throw std::runtime_error("Failed to create Modbus RTU context");
			}
			modbus_set_slave(_modbusContext, 1); // 设置从站ID为1
			_baseAddress = cfg.baseAddress;
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
			/*if (!isConnected()) {
				return false;
			}*/
			data.resize(quantity);
			int result = modbus_read_registers(_modbusContext, startAddress + _baseAddress, quantity, reinterpret_cast<uint16_t*>(data.data()));
			uint16_t number = static_cast<uint16_t>(data[0]);

			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeRegisters(Address startAddress, const std::vector<RegisterValue>& data)
		{
			/*if (!isConnected()) {
				return false;
			}*/
			int result = modbus_write_registers(_modbusContext, startAddress + _baseAddress, data.size(), reinterpret_cast<const uint16_t*>(data.data()));
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeRegister(Address startAddress, RegisterValue32 data, Endianness byteOrder)
		{
			/*if (!isConnected()) {
				return false;
			}*/

			// 将 32 位值拆分为两个 16 位寄存器（高 16 位和低 16 位）
			uint16_t high = static_cast<uint16_t>((static_cast<uint32_t>(data) >> 16) & 0xFFFFu);
			uint16_t low = static_cast<uint16_t>(static_cast<uint32_t>(data) & 0xFFFFu);

			uint16_t regs[2];

			// 根据字节序在两个寄存器之间安排顺序
			// 常见约定：BigEndian - 高字在前；LittleEndian - 低字在前
			switch (byteOrder) {
			case Endianness::BigEndian:
				regs[0] = high;
				regs[1] = low;
				break;
			case Endianness::LittleEndian:
				regs[0] = low;
				regs[1] = high;
				break;
			default:
				// 若有其他自定义字节序，回退为 BigEndian 行为
				regs[0] = high;
				regs[1] = low;
				break;
			}

			int result = modbus_write_registers(_modbusContext, startAddress + _baseAddress, 2, regs);
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeRegisters(Address startAddress, const std::vector<RegisterValue32>& data,
			Endianness byteOrder)
		{
			/*if (!isConnected()) {
				return false;
			}*/

			if (data.empty()) {
				return false;
			}

			const size_t count32 = data.size();
			const size_t regCount = count32 * 2; // 每个 32 位值占用两个 16 位寄存器

			std::vector<uint16_t> regs(regCount);

			for (size_t i = 0; i < count32; ++i) {
				uint32_t v = static_cast<uint32_t>(data[i]);
				uint16_t high = static_cast<uint16_t>((v >> 16) & 0xFFFFu);
				uint16_t low = static_cast<uint16_t>(v & 0xFFFFu);

				if (byteOrder == Endianness::BigEndian) {
					regs[i * 2] = high;
					regs[i * 2 + 1] = low;
				}
				else if (byteOrder == Endianness::LittleEndian) {
					regs[i * 2] = low;
					regs[i * 2 + 1] = high;
				}
				else {
					// 其它情况回退为 BigEndian 行为
					regs[i * 2] = high;
					regs[i * 2 + 1] = low;
				}
			}

			int result = modbus_write_registers(_modbusContext, startAddress + _baseAddress, static_cast<int>(regCount), regs.data());
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::readRegister(Address startAddress, RegisterValue32& data, Endianness byteOrder)
		{
			/*if (!isConnected()) {
		return false;
	}*/

			uint16_t regs[2] = { 0, 0 };
			int result = modbus_read_registers(_modbusContext, startAddress + _baseAddress, 2, regs);
			if (result < 0) {
				return false;
			}

			uint32_t value = 0;
			switch (byteOrder) {
			case Endianness::BigEndian:
				// regs[0] = high, regs[1] = low
				value = (static_cast<uint32_t>(regs[0]) << 16) | static_cast<uint32_t>(regs[1]);
				break;
			case Endianness::LittleEndian:
				// regs[0] = low, regs[1] = high
				value = (static_cast<uint32_t>(regs[1]) << 16) | static_cast<uint32_t>(regs[0]);
				break;
			default:
				// 回退为 BigEndian 行为
				value = (static_cast<uint32_t>(regs[0]) << 16) | static_cast<uint32_t>(regs[1]);
				break;
			}

			data = static_cast<RegisterValue32>(value);
			return true;
		}

		bool ModbusDevice::readRegisters(Address startAddress, std::vector<RegisterValue32>& data, Endianness byteOrder)
		{
			/*if (!isConnected()) {
				return false;
			}*/

			// 需要读取多少个 32 位值
			if (data.empty()) {
				return false;
			}

			const size_t count32 = data.size();
			const size_t regCount = count32 * 2; // 每个 32 位值占用两个 16 位寄存器

			std::vector<uint16_t> regs(regCount, 0);
			int result = modbus_read_registers(_modbusContext, startAddress + _baseAddress, static_cast<int>(regCount), regs.data());
			if (result < 0 || result != static_cast<int>(regCount)) {
				return false;
			}

			for (size_t i = 0; i < count32; ++i) {
				uint16_t r0 = regs[i * 2];
				uint16_t r1 = regs[i * 2 + 1];
				uint32_t value = 0;

				switch (byteOrder) {
				case Endianness::BigEndian:
					// regs[0] = high, regs[1] = low
					value = (static_cast<uint32_t>(r0) << 16) | static_cast<uint32_t>(r1);
					break;
				case Endianness::LittleEndian:
					// regs[0] = low, regs[1] = high
					value = (static_cast<uint32_t>(r1) << 16) | static_cast<uint32_t>(r0);
					break;
				default:
					// 回退为 BigEndian 行为
					value = (static_cast<uint32_t>(r0) << 16) | static_cast<uint32_t>(r1);
					break;
				}

				data[i] = static_cast<RegisterValue32>(value);
			}

			return true;
		}

		bool ModbusDevice::readCoils(Address startAddress, Quantity quantity, std::vector<bool>& data)
		{
			/*if (!isConnected()) {
				return false;
			}*/
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
			/*if (!isConnected()) {
				return false;
			}*/
			int result = modbus_write_bit(_modbusContext, address + _baseAddress, state ? 1 : 0);
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeCoils(Address startAddress, const std::vector<bool>& states)
		{
			/*if (!isConnected()) {
				return false;
			}*/
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
			/*if (!isConnected()) {
				return false;
			}*/
			data.resize(quantity);
			int result = modbus_read_registers(_modbusContext, address, quantity, reinterpret_cast<uint16_t*>(data.data()));
			if (result < 0) {
				return false;
			}
			return true;
		}

		bool ModbusDevice::writeRegistersAbsolute(Address address, const std::vector<RegisterValue>& data)
		{
			/*if (!isConnected()) {
				return false;
			}*/
			int result = modbus_write_registers(_modbusContext, address, data.size(), reinterpret_cast<const uint16_t*>(data.data()));
			if (result < 0) {
				return false;
			}
			return true;
		}
	}
}