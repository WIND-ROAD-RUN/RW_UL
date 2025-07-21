#pragma once

#include <any>
#include <QString>

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
		using ClassIdName = QString;
		using RunTime = unsigned long long;

	}
}
