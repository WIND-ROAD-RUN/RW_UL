#pragma once

#include"imgPro_ImageProcessUtilty.hpp"

namespace rw
{
	namespace imgPro
	{
		struct IndexFunc
		{
			static ProcessResultIndexMap getIndex(const ProcessResult& info);

			//对于index的数值删除符合条件的index，然后返回删除后的index
			static std::vector<ProcessResultIndex> removeIndicesIf(
				ProcessResultIndexMap& indexMap,
				const std::function<bool(ClassId, ProcessResultIndex)>& predicate);
			// 根据lambda表达式删除满足条件的index，并返回被删除的index
			static std::vector<ProcessResultIndex> removeIndicesIfByInfo(
				ProcessResultIndexMap& indexMap,
				const ProcessResult& info,
				const std::function<bool(const rw::DetectionRectangleInfo&)>& predicate);
		};

	}
}
