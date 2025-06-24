#pragma once

#include <string>

namespace rw {
	namespace cla {
		struct EncryptConfig
		{
		public:
			std::string encryptString{};
		public: 
			std::string key{};
		};


		struct DecryptConfig
		{
		public:
			std::string decryptString{ };
		public:
			std::string key{};
		};


	
	}
}
