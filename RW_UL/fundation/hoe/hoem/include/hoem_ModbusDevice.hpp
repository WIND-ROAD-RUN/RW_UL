#pragma once

#include"hoem_utilty.hpp"

typedef struct _modbus modbus_t;

namespace rw
{
	namespace hoem
	{
		//TODO:在连接之上做“优先级调度 + 串行化执行”。核心思路：
		/*	•	单设备单连接，避免底层 PLC 的串行处理被多连接打爆或出现状态竞争。
			•	在同一设备上引入“请求队列”，按优先级（高 / 低）和截止时间（deadline / timeout）调度。
			•	由一个专门的工作线程消费队列，严格串行地对 _modbusContext 执行读写。
			•	高频写入线程将请求作为低优先级入队；重要实时写入线程用高优先级入队，且可带超时与取消。
			•	支持可选的“抢占”：当有高优先级请求到达，工作线程优先处理它（比如用 std::priority_queue）。
			•	若设备确实允许多连接且厂商文档确认并发安全，你可以为“高优先级通道”单独开一个连接，但要确保寄存器级别没有竞态冲突；一般PLC不建议这么做。
			*/

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