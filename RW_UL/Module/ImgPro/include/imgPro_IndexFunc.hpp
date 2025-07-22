#pragma once

#include"imgPro_ImageProcessUtilty.hpp"

namespace rw
{
	namespace imgPro
	{
		using RemoveIndicesIf = std::function<bool(ClassId, ProcessResultIndex)>;
		using RemoveIndicesIfByInfo = std::function<bool(const rw::DetectionRectangleInfo&)>;

		struct IndexGetContext
		{
		public:
			RemoveIndicesIfByInfo removeIndicesIfByInfo;
			RemoveIndicesIf  removeIndicesIf;
		public:
			std::vector<ProcessResultIndex> removedIndices;
			std::vector<ProcessResultIndex> removedIndicesByInfo;
		};

		struct IndexFunc
		{
			static ProcessResultIndexMap getIndex(const ProcessResult& info);

			//对于index的数值删除符合条件的index，然后返回删除后的index
			static std::vector<ProcessResultIndex> removeIndicesIf(
				ProcessResultIndexMap& indexMap,
				const RemoveIndicesIf& predicate);
			// 根据lambda表达式删除满足条件的index，并返回被删除的index
			static std::vector<ProcessResultIndex> removeIndicesIfByInfo(
				ProcessResultIndexMap& indexMap,
				const ProcessResult& info,
				const RemoveIndicesIfByInfo& predicate);
		};

	}
}
