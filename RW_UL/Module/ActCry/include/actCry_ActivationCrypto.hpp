#pragma once

#include "actCry_utility.hpp"

#include"actCry_HWID.hpp"

namespace rw
{
	namespace actCry
	{
		class ActivationCrypto;

		struct ActivationCryptoContext
		{
			friend ActivationCrypto;
		public:
			std::string productName;
		private:
			std::string hwid;
		};

		class ActivationCrypto
		{
		private:
			ActivationCryptoContext _context{};
		public:
			const ActivationCryptoContext& getContext() const
			{
				return _context;
			}

			ActivationCryptoContext& context()
			{
				return _context;
			}
		public:
			bool hwidVerify(const HWIDGenerateCfg & hwidCfg, const HWIDSaveRegistryCfg& cfg);
		private:
			bool hwidVerify();
			bool checkActivationCodeValid();
			bool inputActivationCode();
		public:
			bool operator()();

		};
	}
}


