#pragma once

#include"cla_utility.hpp"
#include <string>

namespace rw {
	namespace cla {
		struct SymmetricEncryptorContext
		{
		public:
			ActivationBits bits;
		public:
			std::string decryptString{ };
		public:
			std::string encryptString{};
		public:
			std::string key{};
		};
	}
}
