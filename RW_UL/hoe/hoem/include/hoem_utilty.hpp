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

		enum class ModbusI
		{
			X00,
			X01,
			X02,
			X03,
			X04,
			X05,
			X06,
			X07,
			X08,
			X09,
			X0A,
			X0B,
			X0C,
			X0D,
			X0E,
			X0F,
			X10,
			X11,
			X12,
			X13,
			X14,
			X15,
			X16,
			X17,
			X18,
			X19,
			X1A,
			X1B,
			X1C,
			X1D,
			X1E,
			X1F
		};

		enum class ModbusO
		{
			Y00,
			Y01,
			Y02,
			Y03,
			Y04,
			Y05,
			Y06,
			Y07,
			Y08,
			Y09,
			Y0A,
			Y0B,
			Y0C,
			Y0D,
			Y0E,
			Y0F,
			Y10,
			Y11,
			Y12,
			Y13,
			Y14,
			Y15,
			Y16,
			Y17,
			Y18,
			Y19,
			Y1A,
			Y1B,
			Y1C,
			Y1D,
			Y1E,
			Y1F
		};
	}
}