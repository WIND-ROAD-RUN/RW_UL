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
	}
}