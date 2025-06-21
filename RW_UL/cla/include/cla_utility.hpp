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

		struct ActivationBitsConvert
		{
			static ActivationString switchBinaryTOHex(const ActivationString& str);
			static ActivationString switchHexTOBinary(const ActivationString& str);
			static ActivationString switchOctTOBinary(const ActivationString& str);
			static ActivationString switchBinaryTOOct(const ActivationString& str);
			static ActivationString switchOctTOHex(const ActivationString& str);
			static ActivationString switchHexTOOct(const ActivationString& str);
		}; 

	}
}
