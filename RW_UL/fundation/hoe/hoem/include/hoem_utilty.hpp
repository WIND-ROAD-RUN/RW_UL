#pragma once

#include <vector>
#include <cstdint>
#include<string>
#include <stdexcept>

namespace rw
{
	namespace hoem
	{
		using Address = uint16_t;
		using Quantity = uint16_t;
		using RegisterValue = uint16_t;

		enum class ModbusDeviceName
		{
			keRuiE
		};

		enum class Endianness
		{
			BigEndian,    // 大端
			LittleEndian  // 小端
		};

		struct ModbusConfig
		{
		public:
			std::string ip{};
			int port{};
		};

		enum class ModbusI
		{
			X00 = 0,
			X01 = 1,
			X02 = 2,
			X03 = 3,
			X04 = 4,
			X05 = 5,
			X06 = 6,
			X07 = 7,
			X08 = 8,
			X09 = 9,
			X0A = 10,
			X0B = 11,
			X0C = 12,
			X0D = 13,
			X0E = 14,
			X0F = 15,
			X10 = 16,
			X11 = 17,
			X12 = 18,
			X13 = 19,
			X14 = 20,
			X15 = 21,
			X16 = 22,
			X17 = 23,
			X18 = 24,
			X19 = 25,
			X1A = 26,
			X1B = 27,
			X1C = 28,
			X1D = 29,
			X1E = 30,
			X1F = 31
		};

		enum class ModbusO
		{
			Y00 = 0,
			Y01 = 1,
			Y02 = 2,
			Y03 = 3,
			Y04 = 4,
			Y05 = 5,
			Y06 = 6,
			Y07 = 7,
			Y08 = 8,
			Y09 = 9,
			Y0A = 10,
			Y0B = 11,
			Y0C = 12,
			Y0D = 13,
			Y0E = 14,
			Y0F = 15,
			Y10 = 16,
			Y11 = 17,
			Y12 = 18,
			Y13 = 19,
			Y14 = 20,
			Y15 = 21,
			Y16 = 22,
			Y17 = 23,
			Y18 = 24,
			Y19 = 25,
			Y1A = 26,
			Y1B = 27,
			Y1C = 28,
			Y1D = 29,
			Y1E = 30,
			Y1F = 31
		};

		inline int32_t swapEndian(int32_t value)
		{
			return ((value & 0xFF000000) >> 24) |
				((value & 0x00FF0000) >> 8) |
				((value & 0x0000FF00) << 8) |
				((value & 0x000000FF) << 24);
		}

		inline int32_t modbusIToInt32(ModbusI modbusI, Endianness endianness)
		{
			int32_t value = 1 << static_cast<int32_t>(modbusI);
			if (endianness == Endianness::BigEndian)
			{
				return swapEndian(value); // 转换为大端表示
			}
			return value; // 小端表示直接返回
		}

		inline int32_t modbusOToInt32(ModbusO modbusO, Endianness endianness)
		{
			int32_t value = 1 << static_cast<int32_t>(modbusO);
			if (endianness == Endianness::BigEndian)
			{
				return swapEndian(value); // 转换为大端表示
			}
			return value; // 小端表示直接返回
		}

		inline std::vector<RegisterValue> toRegisterValues(ModbusI i, Endianness endianness)
		{
			int32_t value = static_cast<int32_t>(i);

			RegisterValue high = static_cast<RegisterValue>((value >> 16) & 0xFFFF); // 高16位
			RegisterValue low = static_cast<RegisterValue>(value & 0xFFFF);         // 低16位

			std::vector<RegisterValue> result;
			if (endianness == Endianness::BigEndian)
			{
				result.push_back(high);
				result.push_back(low);
			}
			else // LittleEndian
			{
				result.push_back(low);
				result.push_back(high);
			}

			return result;
		}

		inline std::vector<RegisterValue> toRegisterValues(ModbusO i, Endianness endianness)
		{
			int32_t value = static_cast<int32_t>(i);

			RegisterValue high = static_cast<RegisterValue>((value >> 16) & 0xFFFF); // 高16位
			RegisterValue low = static_cast<RegisterValue>(value & 0xFFFF);         // 低16位

			std::vector<RegisterValue> result;
			if (endianness == Endianness::BigEndian)
			{
				result.push_back(high);
				result.push_back(low);
			}
			else // LittleEndian
			{
				result.push_back(low);
				result.push_back(high);
			}

			return result;
		}

		inline int32_t fromRegisterValuesToInt32(const std::vector<RegisterValue>& values, Endianness endianness)
		{
			if (values.size() != 2)
			{
				throw std::invalid_argument("Invalid register values size for ModbusI. Expected 2 values.");
			}

			int32_t value = 0;
			if (endianness == Endianness::BigEndian)
			{
				value = (static_cast<int32_t>(values[0]) << 16) | static_cast<int32_t>(values[1]);
			}
			else // LittleEndian
			{
				value = (static_cast<int32_t>(values[1]) << 16) | static_cast<int32_t>(values[0]);
			}

			return value;
		}

		inline ModbusI fromRegisterValuesToModbusI(const std::vector<RegisterValue>& values, Endianness endianness)
		{
			return static_cast<ModbusI>(fromRegisterValuesToInt32(values, endianness));
		}

		inline ModbusO fromRegisterValuesToModbusO(const std::vector<RegisterValue>& values, Endianness endianness)
		{
			return static_cast<ModbusO>(fromRegisterValuesToInt32(values, endianness));
		}
	}
}