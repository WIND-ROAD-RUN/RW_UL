#pragma once

#include "imgPro_EliminationInfoFunc.hpp"
#include"imgPro_ImageProcessUtilty.hpp"

namespace rw
{
	namespace imgPro
	{
		struct DefectResultInfo
		{
		public:
			bool isBad{};
		public:
			std::unordered_map<ClassId, std::vector<EliminationItem>> defects;
			std::unordered_map<ClassId, std::vector<EliminationItem>> disableDefects;
		};

		struct DefectResultInfoFunc
		{
		public:
			struct DefectResultGetConfig
			{
			public:
				bool isEnable{ false };
			};
			using ClassIdWithConfigMap = std::unordered_map<ClassId, DefectResultGetConfig>;
		public:
			static DefectResultInfo getDefectResultInfo(const EliminationInfo& eliminationInfo, const ClassIdWithConfigMap& config);
		};
	}
}
