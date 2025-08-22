#include"imgPro_DefectDrawFunc.hpp"

#include "imgPro_DefectResultInfoFunc.hpp"
#include"imgPro_ImagePainter.hpp"

namespace rw
{
	namespace imgPro
	{
		void DefectDrawFunc::ConfigDefectDraw::setAllIdsWithSameColor(const std::vector<ClassId>& ids, Color color,
			bool isGood)
		{
			for (const auto& id : ids)
			{
				if (isGood)
				{
					classIdWithColorWhichIsGood[id] = color;
				}
				else
				{
					classIdWithColorWhichIsBad[id] = color;
				}
			}
		}

		void DefectDrawFunc::drawDefectRecs(QImage& img, const DefectResultInfo& info,
			const ProcessResult& processResult, const ConfigDefectDraw& config)
		{
			if (img.isNull() || processResult.empty()) {
				return; 
			}

			if (config.isDrawDefects)
			{
				drawDefectGroup(
					img,
					info.defects,
					processResult,
					config,
					config.classIdWithColorWhichIsBad,
					Color::Red
				);
			}

			if (config.isDrawDisableDefects)
			{
				drawDefectGroup(
					img,
					info.disableDefects,
					processResult,
					config,
					config.classIdWithColorWhichIsGood,
					Color::Green
				);
			}
		}

		void DefectDrawFunc::drawRunText(QImage& img, const ConfigRunText& config)
		{
			QVector<QString> textList;
			std::vector<Color> configList;
			auto textColor = config.operatorTimeTextColor;
			if (config.isDisOperatorTime)
			{
				configList.push_back(textColor);
				textList.push_back(config.operatorTimeText);
			}
			textColor = config.processImgTimeTextColor;
			if (config.isDisProcessImgTime)
			{
				configList.push_back(textColor);
				textList.push_back(config.processImgTimeText);
			}

			if (config.isDrawExtraText)
			{
				textColor = config.extraTextColor;
				configList.push_back(textColor);
				for (const auto& item : config.extraTexts)
				{
					textList.push_back(item);
				}
			}

			rw::imgPro::ImagePainter::drawTextOnImage(img, textList, configList, config.runTextProportion);
		}

		void DefectDrawFunc::drawDefectGroup(
			QImage& img,
			const std::unordered_map<ClassId, std::vector<EliminationItem>>& group, 
			const ProcessResult& processResult,
			const DefectDrawFunc::ConfigDefectDraw& config, 
			const std::unordered_map<ClassId, Color>& colorMap,
			Color defaultColor
		)
		{
			rw::imgPro::ConfigDrawRect recCfg;
			recCfg.fontSize = config.fontSize;
			recCfg.thickness = config.thickness;
			recCfg.textLocate = config.textLocate;
			recCfg.isRegion = config.isDrawMask;
			recCfg.alpha = config.alpha;
			recCfg.thresh = config.thresh;
			recCfg.maxVal = config.maxVal;
			recCfg.hasFrame = config.hasFrame;
			recCfg.rectColor = defaultColor;
			recCfg.textColor = defaultColor;

			for (const auto& pairs : group)
			{
				if (config.classIdIgnoreDrawSet.find(pairs.first) != config.classIdIgnoreDrawSet.end())
				{
					continue;
				}
				QString processTextPre = (config.classIdNameMap.find(pairs.first) != config.classIdNameMap.end()) ?
					config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);

				for (const auto& item : pairs.second)
				{
					recCfg.text.clear();
					if (config.isDisScoreText)
					{
						recCfg.text = processTextPre + " : " + QString::number(item.score, 'f', config.scoreDisPrecision);
					}
					if (config.isDisAreaText)
					{
						recCfg.text = recCfg.text + " , " + QString::number(item.area, 'f', config.areaDisPrecision);
					}

					auto findColor = colorMap.find(pairs.first);
					if (findColor != colorMap.end())
					{
						recCfg.rectColor = findColor->second;
						recCfg.textColor = findColor->second;
					}
					else
					{
						recCfg.rectColor = defaultColor;
						recCfg.textColor = defaultColor;
					}

					auto& proResult = processResult[item.index];

					if (proResult.segMaskValid && config.isDrawMask)
					{
						rw::imgPro::ConfigDrawMask maskCfg;
						maskCfg.alpha = config.alpha;
						maskCfg.thresh = config.thresh;
						maskCfg.maxVal = config.maxVal;
						maskCfg.maskColor = recCfg.rectColor;
						maskCfg.hasFrame = config.hasFrame;
						maskCfg.rectCfg = recCfg;
						maskCfg.rectCfg.rectColor = recCfg.rectColor;
						maskCfg.rectCfg.isRegion = false;
						rw::imgPro::ImagePainter::drawMaskOnSourceImg(img, proResult, maskCfg);
					}
					else
					{
						rw::imgPro::ImagePainter::drawShapesOnSourceImg(img, proResult, recCfg);
					}
				}
			}
		}
	}
}