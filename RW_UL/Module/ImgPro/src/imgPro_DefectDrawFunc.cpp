#include"imgPro_DefectDrawFunc.hpp"

#include "imgPro_DefectResultInfoFunc.hpp"
#include"imgPro_ImagePainter.hpp"

namespace rw
{
	namespace imgPro
	{
		void DefectDrawFunc::DefectDrawConfig::setAllIdsWithSameColor(const std::vector<ClassId>& ids, Color color,
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
			const ProcessResult& processResult, const DefectDrawConfig& config)
		{
			if (img.isNull() || processResult.empty()) {
				return; // 无效图像或结果
			}

			rw::imgPro::ConfigDrawRect recCfg;
			recCfg.fontSize = config.fontSize;
			recCfg.thickness = config.thickness;
			recCfg.textLocate = config.textLocate;
			recCfg.isRegion = config.isDrawMask;
			recCfg.alpha = config.alpha;
			recCfg.thresh = config.thresh;
			recCfg.maxVal = config.maxVal;
			recCfg.hasFrame = config.hasFrame;
			if (config.isDrawDefects)
			{
				recCfg.rectColor = Color::Red;
				recCfg.textColor = Color::Red;
				for (const auto& pairs : info.defects)
				{
					QString processTextPre = (config.classIdNameMap.find(pairs.first) != config.classIdNameMap.end()) ?
						config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
					for (const auto& item : pairs.second)
					{
						if (config.isDisScoreText)
						{
							recCfg.text =
								processTextPre + " : " + QString::number(item.score, 'f', 1);
						}
						if (config.isDisAreaText)
						{
							recCfg.text =
								recCfg.text + " , " + QString::number(item.area, 'f', 2);
						}

						auto findColor = config.classIdWithColorWhichIsBad.find(pairs.first);
						if (findColor != config.classIdWithColorWhichIsBad.end())
						{
							recCfg.rectColor = findColor->second;
							recCfg.textColor = findColor->second;
						}

						auto & proResult = processResult[item.index];

						if (proResult.segMaskValid&& config.isDrawMask)
						{
							rw::imgPro::ConfigDrawMask maskCfg;
							maskCfg.alpha = config.alpha;
							maskCfg.thresh = config.thresh;
							maskCfg.maxVal = config.maxVal;
							maskCfg.maskColor = findColor->second;
							maskCfg.hasFrame = config.hasFrame;
							maskCfg.rectCfg = recCfg;
							maskCfg.rectCfg.rectColor = findColor->second;
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

			if (config.isDrawDisableDefects)
			{
				recCfg.rectColor = Color::Green;
				recCfg.textColor = Color::Green;

				for (const auto& pairs : info.disableDefects)
				{
					QString processTextPre = (config.classIdNameMap.find(pairs.first) != config.classIdNameMap.end()) ?
						config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
					for (const auto& item : pairs.second)
					{
						if (config.isDisScoreText)
						{
							recCfg.text =
								processTextPre + " : " + QString::number(item.score, 'f', 2);
						}
						if (config.isDisAreaText)
						{
							recCfg.text =
								recCfg.text + " , " + QString::number(item.area, 'f', 1);
						}

						auto findColor = config.classIdWithColorWhichIsGood.find(pairs.first);
						if (findColor != config.classIdWithColorWhichIsGood.end())
						{
							recCfg.rectColor = findColor->second;
							recCfg.textColor = findColor->second;
						}

						auto& proResult = processResult[item.index];

						if (proResult.segMaskValid && config.isDrawMask)
						{
							rw::imgPro::ConfigDrawMask maskCfg;
							maskCfg.alpha = config.alpha;
							maskCfg.thresh = config.thresh;
							maskCfg.maxVal = config.maxVal;
							maskCfg.maskColor = findColor->second;
							maskCfg.hasFrame = config.hasFrame;
							maskCfg.rectCfg = recCfg;
							maskCfg.rectCfg.rectColor = findColor->second;
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

		void DefectDrawFunc::drawRunText(QImage& img, const RunTextConfig& config)
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
	}
}