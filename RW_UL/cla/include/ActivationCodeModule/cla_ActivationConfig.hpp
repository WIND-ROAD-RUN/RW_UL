#pragma once

#include<map>
#include<string>

namespace rw
{
	namespace cla
	{
		using ConfigName = std::string;
		using ConfigValue = std::string;

		struct ActivationConfig
		{
		public:
			std::map<ConfigName, ConfigValue> configs;
		};
	}
}
