#pragma once

#include"actCry_utility.hpp"
#include <chrono>
#include <string>
#include <cstddef>

namespace rw
{
	namespace actCry
	{

		using ActivationCode = std::string;

		struct ActivationInfoRegistryCfg
		{
			std::string name;
			std::string key;
			std::string keyPath{ "Software\\RW\\ActCryActivationCode" };
		};

		class ActivationInfo
		{
		public:
			std::string hwid;
			std::chrono::system_clock::time_point startTime{};
			std::chrono::system_clock::time_point endTime{};
		public:
			static bool save(const std::string& hwid, const ActivationInfo& info,const ActivationInfoRegistryCfg & cfg);
			static std::string load(const ActivationInfo& cfg);

		public:
			static ActivationCode generateActivationCode(const ActivationInfo& info,const std::string & key);
			static ActivationInfo parseActivationCode(const ActivationCode& code, const std::string& key,bool & isOk);
		};
	}

}