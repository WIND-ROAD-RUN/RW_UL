#pragma once

#include"cla_utility.hpp"
#include <string>

namespace rw {
	namespace cla {
		struct SymmetricEncryptorContext
		{
		public:
			ActivationCodeStruct decryptString{ };
		public:
			ActivationCodeStruct encryptString{};
		public:
			std::string key{};
		};
	}
}
