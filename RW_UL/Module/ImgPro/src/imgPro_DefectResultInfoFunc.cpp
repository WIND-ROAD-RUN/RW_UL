#include"imgPro_DefectResultInfoFunc.hpp"
namespace rw
{
	namespace imgPro
	{
		DefectResultInfo DefectResultInfoFunc::getDefectResultInfo(const EliminationInfo& eliminationInfo,
			const ClassIdWithConfigMap& config)
		{
			return getDefectResultInfo(eliminationInfo, config, GetDefectResultExtraOperateWhichIsDefects{}, GetDefectResultExtraOperateWhichIsDisableDefects{});
		}

		DefectResultInfo DefectResultInfoFunc::getDefectResultInfo(const EliminationInfo& eliminationInfo,
			const ClassIdWithConfigMap& config, const GetDefectResultExtraOperateWhichIsDefects& getDefectResultExtraOperate,
			const GetDefectResultExtraOperateWhichIsDisableDefects& getDefectResultExtraOperateDisable
		)
		{
			DefectResultInfo result;
			result.isBad = false;

			for (const auto& kv : config)
			{
				ClassId classId = kv.first;
				const Config& cfg = kv.second;

				auto it = eliminationInfo.defectItems.find(classId);
				if (it != eliminationInfo.defectItems.end())
				{
					const auto& items = it->second;

					for (const auto& item : items)
					{
						if (cfg.isEnable)
						{
							if (item.isBad)
							{
								result.isBad = true;
								if (getDefectResultExtraOperate)
								{
									getDefectResultExtraOperate(item);
								}
								result.defects[classId].push_back(item);
							}
							else
							{
								if (getDefectResultExtraOperateDisable)
								{
									getDefectResultExtraOperateDisable(item);
								}
								result.disableDefects[classId].push_back(item);
							}
						}
						else
						{
							if (getDefectResultExtraOperateDisable)
							{
								getDefectResultExtraOperateDisable(item);
							}
							result.disableDefects[classId].push_back(item);
						}
					}
				}
			}

			return result;
		}

		DefectResultInfo DefectResultInfoFunc::getDefectResultInfo(const ProcessResult& processResult,
			const ClassIdWithEliminationInfoConfigMap& classIdWithEliminationInfoConfigMap,
			const EliminationInfo& eliminationInfo, 
			const ClassIdWithConfigMap& config,
			const GetDefectResultExtraOperateWhichIsDefects& getDefectResultExtraOperate,
			const GetDefectResultExtraOperateWhichIsDisableDefects& getDefectResultExtraOperateDisable,
			const GetDefectResultExtraOperateWithFullInfo& getDefectResultExtraOperateWithFullInfo)
		{
			DefectResultInfo result;
			result.isBad = false;

			for (const auto& kv : config)
			{
				ClassId classId = kv.first;
				const Config& cfg = kv.second;

				auto it = eliminationInfo.defectItems.find(classId);
				auto itEliCfg = classIdWithEliminationInfoConfigMap.find(classId);
				if (it != eliminationInfo.defectItems.end())
				{
					const auto& items = it->second;

					for (const auto& item : items)
					{
						if (cfg.isEnable)
						{
							if (item.isBad)
							{
								result.isBad = true;
								if (getDefectResultExtraOperate)
								{
									getDefectResultExtraOperate(item);
								}
								result.defects[classId].push_back(item);
							}
							else
							{
								if (getDefectResultExtraOperateDisable)
								{
									getDefectResultExtraOperateDisable(item);
								}
								result.disableDefects[classId].push_back(item);
							}
						}
						else
						{
							if (getDefectResultExtraOperateDisable)
							{
								getDefectResultExtraOperateDisable(item);
							}
							result.disableDefects[classId].push_back(item);
						}

						if (itEliCfg!= classIdWithEliminationInfoConfigMap.end())
						{
							if (getDefectResultExtraOperateWithFullInfo)
							{
								getDefectResultExtraOperateWithFullInfo(
									processResult[item.index],
									itEliCfg->second,
									item,
									cfg
								);
							}
						}
					}
				}
			}

			return result;
		}
	}
}