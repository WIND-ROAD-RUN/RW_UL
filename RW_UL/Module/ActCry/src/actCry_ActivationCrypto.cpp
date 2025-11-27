#include"actCry_ActivationCrypto.hpp"

#include<stdexcept>

namespace rw
{
	namespace actCry
	{

		bool ActivationCrypto::hwidVerify(const HWIDGenerateCfg& hwidCfg, const HWIDSaveRegistryCfg& cfg)
		{
			if (_context.productName.empty())
			{
				throw std::runtime_error("ActivationCrypto::hwidVerify: productName is empty");
			}

			auto hwid=HWID::load(cfg);
			if (hwid.empty())
			{
				hwid = HWID::generate(hwidCfg);
				auto saveResult=HWID::save(hwid, cfg);
				if (!saveResult)
				{
					return false;
				}
			}

			// verify HWID is changed
			auto currentHwid = HWID::generate(hwidCfg);
			if (hwid != currentHwid)
			{
				return false;
			}

			_context.hwid = hwid;
			return true;
		}

		bool ActivationCrypto::hwidVerify()
		{
			HWIDGenerateCfg hwidCfg;
			hwidCfg.productName = _context.productName;
			HWIDSaveRegistryCfg saveCfg;
			saveCfg.name = _context.productName;
			auto hwidVerifyResult = hwidVerify(hwidCfg, saveCfg);
			return hwidVerifyResult;
		}

		bool ActivationCrypto::checkActivationCodeValid()
		{
			return false;
		}

		bool ActivationCrypto::inputActivationCode()
		{
			bool result{true};
			if (_context.inputActivationCodeFunc)
			{
				_context.inputActivationCode = _context.inputActivationCodeFunc(result);

			}
			return result;
		}

		bool ActivationCrypto::operator()()
		{
			auto hwidVerifyResult = hwidVerify();
			if (!hwidVerifyResult)
			{
				return false;
			}

			auto isActivationCodeValid = checkActivationCodeValid();

			if (!isActivationCodeValid)
			{
				auto inputActivationCodeResult = inputActivationCode();
				if (!inputActivationCodeResult)
				{
					return false;
				}
			}

			return hwidVerifyResult;
		}
	}
}
