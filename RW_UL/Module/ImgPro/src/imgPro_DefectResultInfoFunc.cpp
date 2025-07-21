#include"imgPro_DefectResultInfoFunc.hpp"
namespace rw
{
	namespace imgPro
	{
		DefectResultInfo DefectResultInfoFunc::getDefectResultInfo(const EliminationInfo& eliminationInfo,
			const ClassIdWithConfigMap& config)
		{
			DefectResultInfo result;
			result.isBad = false;

			for (const auto& kv : config)
			{
				ClassId classId = kv.first;
				const DefectResultGetConfig& cfg = kv.second;

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
								result.defects[classId].push_back(item);
							}
							else
							{
								result.disableDefects[classId].push_back(item);
							}
						}
						else
						{
							result.disableDefects[classId].push_back(item);
						}
					}
				}
			}

			return result;
		}
	}
}
