#pragma once

#include<QString>

namespace rw
{
	namespace rqw
	{
		enum class WarningType
		{
			Warning,
			Error,
			Info
		};

		struct WarningInfo
		{
		public:
			WarningType type{ WarningType::Info };
			QString message{""};
		};

	}
}
