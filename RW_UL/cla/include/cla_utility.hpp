#pragma once

#include <string>

namespace rw
{
	namespace cla
	{
		using UniqueIdentifier = std::string;
		using ActivationString = std::string;

		enum class ActivationBits
		{
			Hexadecimal,
			Octal,
			Binary
		};

	}
}
