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

		struct DefectResultGetConfig
		{
		public:
			bool isEnable{ false };
		};

		using GetDefectResultExtraOperateWhichIsDefects = 
			std::function<void(
				const EliminationItem&
				)>;

		using GetDefectResultExtraOperateWhichIsDisableDefects = 
			std::function<void(
			const EliminationItem&
			)>;

		using GetDefectResultExtraOperateWithFullInfo = 
			std::function<void(
			const rw::DetectionRectangleInfo&,
			const EliminationInfoGetConfig &,
			const EliminationItem&,
			const DefectResultGetConfig&
			)>;

		struct DefectResultGetContext
		{
		public:
			GetDefectResultExtraOperateWhichIsDefects getDefectResultExtraOperate{};
			GetDefectResultExtraOperateWhichIsDisableDefects getDefectResultExtraOperateDisable{};
		};


		struct DefectResultInfoFunc
		{
		public:
			using Config = DefectResultGetConfig;
			using ClassIdWithConfigMap = std::unordered_map<ClassId, Config>;
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