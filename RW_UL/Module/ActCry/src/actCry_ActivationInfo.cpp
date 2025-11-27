#include "actCry_ActivationInfo.hpp"

#include <iomanip>

#include"cla_StrEncryption.hpp"


namespace rw
{
	namespace actCry
	{
		static cla::StrConfig ActivationInfoToStrConfig(const ActivationInfo& info)
		{
			cla::StrConfig result{};
			result.insert(info.hwid, info.hwid);
			result.insert("startTime", std::to_string(std::chrono::duration_cast<std::chrono::seconds>(info.startTime.time_since_epoch()).count()));
			result.insert("endTime", std::to_string(std::chrono::duration_cast<std::chrono::seconds>(info.endTime.time_since_epoch()).count()));
			return result;
		}


		bool ActivationInfo::save(const std::string& hwid, const ActivationInfo& info,
			const ActivationInfoRegistryCfg& cfg)
		{
			ActivationCode code = generateActivationCode(info, cfg.key);

		}

		ActivationCode ActivationInfo::generateActivationCode(const ActivationInfo& info, const std::string& key)
		{
			cla::StrConfig config = ActivationInfoToStrConfig(info);
			return cla::StrEncryption::StrConfigToHex(config, key);
		}

		ActivationInfo ActivationInfo::parseActivationCode(const ActivationCode& code, const std::string& key, bool& isOk)
		{
			ActivationInfo result{};
			isOk = false;
			bool parseOk = false;
			cla::StrConfig config = cla::StrEncryption::StrConfigFromHex(code, key, parseOk);
			if (!parseOk)
			{
				return result;
			}
			result.hwid = config.deserialize(config).configs.begin()->first;
			try
			{
				auto startTimeStr = config.deserialize(config).configs.at("startTime");
				auto endTimeStr = config.deserialize(config).configs.at("endTime");
				auto startTimeSec = std::stoll(startTimeStr);
				auto endTimeSec = std::stoll(endTimeStr);
				result.startTime = std::chrono::system_clock::time_point{ std::chrono::seconds{startTimeSec} };
				result.endTime = std::chrono::system_clock::time_point{ std::chrono::seconds{endTimeSec} };
			}
			catch (const std::exception&)
			{
				return result;
			}
			isOk = true;
			return result;
		}
	}
}