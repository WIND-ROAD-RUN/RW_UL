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


		using GetDefectResultExtraOperateWhichIsDefects = std::function<void(const EliminationItem &)>;
		using GetDefectResultExtraOperateWhichIsDisableDefects = std::function<void(const EliminationItem&)>;

		struct DefectResultGetContext
		{
		public:
			GetDefectResultExtraOperateWhichIsDefects getDefectResultExtraOperate{};
			GetDefectResultExtraOperateWhichIsDisableDefects getDefectResultExtraOperateDisable{};
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

			static DefectResultInfo getDefectResultInfo(
				const EliminationInfo& eliminationInfo,
				const ClassIdWithConfigMap& config,
				const GetDefectResultExtraOperateWhichIsDefects& getDefectResultExtraOperate,
				const GetDefectResultExtraOperateWhichIsDisableDefects&
				getDefectResultExtraOperateDisable
			);
		};
	}
}
