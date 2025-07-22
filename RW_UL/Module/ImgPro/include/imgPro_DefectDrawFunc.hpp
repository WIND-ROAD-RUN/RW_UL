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
			public:
				int fontSize{ 30 };
				int thickness{3};
			};

			struct RunTextConfig
			{
			public:
				bool isDrawExtraText{ true };
				QVector<QString> extraTexts{};
				rw::rqw::RQWColor extraTextColor{ rw::rqw::RQWColor::Red };
			public:
				bool isDisProcessImgTime{ true };
				QString processImgTimeText{};
				rw::rqw::RQWColor processImgTimeTextColor{ rw::rqw::RQWColor::Blue };
			public:
				bool isDisOperatorTime{ true };
				QString operatorTimeText{};
				rw::rqw::RQWColor operatorTimeTextColor{ rw::rqw::RQWColor::Green };
			};
		public:
			static void drawDefectRecs(QImage& img, const DefectResultInfo& info, const ProcessResult& processResult, const DefectDrawConfig& config);
			static void drawRunText(QImage& img,const RunTextConfig & config);
		};
	}
}
