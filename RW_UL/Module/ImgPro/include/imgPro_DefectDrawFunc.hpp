#pragma once

#include "imgPro_DefectResultInfoFunc.hpp"
#include"imgPro_ImageProcessUtilty.hpp"

#include"rqw_rqwColor.hpp"

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
			public:
				std::unordered_map<ClassId, ClassIdName> classIdNameMap;
			public:
				bool isDisScoreText{true};
				bool isDisAreaText{ true };
			public:
				std::unordered_map<ClassId, rw::rqw::RQWColor> classIdWithColorWhichIsGood;
				std::unordered_map<ClassId, rw::rqw::RQWColor> classIdWithColorWhichIsBad;
			public:
				void setAllIdsWithSameColor(const std::vector<ClassId>& ids,rw::rqw::RQWColor color,bool isGood);
			};

			struct RunTextConfig
			{
			public:
				bool isDrawExtraText{ true };
				QVector<QString> extraTexts{};
			public:
				bool isDisProcessImgTime{ true };
				QString processImgTimeText{};
			public:
				bool isDisOperatorTime{ true };
				QString operatorTimeText{};
			};
		public:
			static void drawDefectRecs(QImage& img, const DefectResultInfo& info, const ProcessResult& processResult, const DefectDrawConfig& config);
			static void drawRunText(QImage& img,const RunTextConfig & config);
		};
	}
}
