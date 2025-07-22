#pragma once

#include "imgPro_DefectResultInfoFunc.hpp"
#include"imgPro_ImageProcessUtilty.hpp"

#include"rqw_rqwColor.hpp"

#include<QImage>

#include "imgPro_ImagePainter.hpp"


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
				std::unordered_map<ClassId, Color> classIdWithColorWhichIsGood;
				std::unordered_map<ClassId, Color> classIdWithColorWhichIsBad;
			public:
				void setAllIdsWithSameColor(const std::vector<ClassId>& ids, Color color,bool isGood);
			public:
				int fontSize{ 30 };
				int thickness{3};
			public:
				ConfigDrawRect::TextLocate textLocate{ ConfigDrawRect::TextLocate::LeftTopOut };
			};

			struct RunTextConfig
			{
			public:
				bool isDrawExtraText{ true };
				QVector<QString> extraTexts{};
				Color extraTextColor{ Color::Red };
			public:
				bool isDisProcessImgTime{ true };
				QString processImgTimeText{};
				Color processImgTimeTextColor{ Color::Blue };
			public:
				bool isDisOperatorTime{ true };
				QString operatorTimeText{};
				Color operatorTimeTextColor{ Color::Green };
			public:
				double runTextProportion = 0.06;
			};
		public:
			static void drawDefectRecs(QImage& img, const DefectResultInfo& info, const ProcessResult& processResult, const DefectDrawConfig& config);
			static void drawRunText(QImage& img,const RunTextConfig & config);
		};
	}
}
