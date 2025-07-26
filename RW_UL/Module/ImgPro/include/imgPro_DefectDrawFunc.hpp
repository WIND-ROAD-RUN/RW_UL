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
				bool isDisScoreText{ true };
				bool isDisAreaText{ true };
			public:
				std::unordered_map<ClassId, Color> classIdWithColorWhichIsGood;
				std::unordered_map<ClassId, Color> classIdWithColorWhichIsBad;
			public:
				void setAllIdsWithSameColor(const std::vector<ClassId>& ids, Color color, bool isGood);
			public:
				int fontSize{ 30 };
				int thickness{ 3 };
			public:
				ConfigDrawRect::TextLocate textLocate{ ConfigDrawRect::TextLocate::LeftTopOut };
			public:
				bool isDrawMask{ false };
				double alpha{ 0.3 };
				double thresh{ 0.5 };
				double maxVal{ 1.0 };
				bool hasFrame{ true };
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
			static void drawRunText(QImage& img, const RunTextConfig& config);
		private:
			static void drawDefectGroup(
				QImage& img,
				const std::unordered_map<ClassId, std::vector<EliminationItem>>& group,
				const ProcessResult& processResult,
				const DefectDrawFunc::DefectDrawConfig& config,
				const std::unordered_map<ClassId, Color>& colorMap,
				Color defaultColor,
				int scorePrecision,
				int areaPrecision);
            
		};
	}
}