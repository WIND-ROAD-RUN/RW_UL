#pragma once

#include "imgPro_DefectResultInfoFunc.hpp"
#include"imgPro_ImageProcessUtilty.hpp"
#include<QImage>

namespace rw
{
	namespace imgPro
	{
		struct DefectDrawFunc
		{
		public:
			struct DefectDrawConfig
			{
			public:
				bool isDrawDefects{ true };
				bool isDrawDisableDefects{ true };
				std::unordered_map<ClassId, ClassIdName> classIdNameMap;
			};
		public:
			static void drawDefectRecs(QImage& img, const DefectResultInfo& info, const ProcessResult& processResult, const DefectDrawConfig& config);

		};
	}
}
