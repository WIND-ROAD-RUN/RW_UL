#pragma once

#include <any>

#include"ime_utilty.hpp"
#include <unordered_map>

namespace rw
{
	namespace imgPro
	{
		using ClassId = size_t;
		using ProcessResult = std::vector<rw::DetectionRectangleInfo>;
		using ProcessResultIndex = size_t;
		using ProcessResultIndexMap = std::unordered_map<ClassId, std::set<ProcessResultIndex>>;

	}
}
