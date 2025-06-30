#pragma once

#include <string>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include"cla_EncrpyUtilty.hpp"

namespace rw
{
	namespace cla
	{
		class SymmetricEncryptor
		{
		private:
			SymmetricEncryptorContext _context;
		public:
			SymmetricEncryptorContext& context() { return _context; }
			const SymmetricEncryptorContext& context() const { return _context; }
		public:
			explicit SymmetricEncryptor(SymmetricEncryptorContext context);
			~SymmetricEncryptor() = default;
		};
	}
}