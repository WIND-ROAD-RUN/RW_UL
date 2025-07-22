#include"imgPro_DefectDrawFunc.hpp"

#include "imgPro_DefectResultInfoFunc.hpp"
#include "rqw_ImagePainter.h"

namespace rw
{
	namespace imgPro
	{
		void DefectDrawFunc::DefectDrawConfig::setAllIdsWithSameColor(const std::vector<ClassId>& ids, rw::rqw::RQWColor color,
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

			if (config.isDrawDefects)
			{
				rw::rqw::ImagePainter::PainterConfig painterConfig;
				painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
				painterConfig.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
				for (const auto& pairs : info.defects)
				{
					QString processTextPre = (config.classIdNameMap.find(pairs.first) != config.classIdNameMap.end()) ?
						config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
					for (const auto& item : pairs.second)
					{
						if (config.isDisScoreText)
						{
							painterConfig.text =
								processTextPre + " : " + QString::number(item.score, 'f', 1);
						}
						if (config.isDisAreaText)
						{
							painterConfig.text =
								painterConfig.text + " , " + QString::number(item.area, 'f', 2);
						}

						auto findColor = config.classIdWithColorWhichIsBad.find(pairs.first);
						if (findColor!= config.classIdWithColorWhichIsBad.end())
						{
							auto color = rqw::RQWColorToQColor(findColor->second);
							painterConfig.color = color;
							painterConfig.textColor = color;
						}
						
						rw::rqw::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
					}
				}
			}

			if (config.isDrawDisableDefects)
			{
				rw::rqw::ImagePainter::PainterConfig painterConfig;
				painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
				painterConfig.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
				for (const auto& pairs : info.disableDefects)
				{
					QString processTextPre = (config.classIdNameMap.find(pairs.first) != config.classIdNameMap.end()) ?
						config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
					for (const auto& item : pairs.second)
					{
						if (config.isDisScoreText)
						{
							painterConfig.text =
								processTextPre + " : " + QString::number(item.score, 'f', 2);
						}
						if (config.isDisAreaText)
						{
							painterConfig.text =
								painterConfig.text + " , " + QString::number(item.area, 'f', 1);
						}

						auto findColor = config.classIdWithColorWhichIsGood.find(pairs.first);
						if (findColor != config.classIdWithColorWhichIsGood.end())
						{
							auto color = rqw::RQWColorToQColor(findColor->second);
							painterConfig.color = color;
							painterConfig.textColor = color;
						}

						rw::rqw::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
					}
				}
			}

		}

		void DefectDrawFunc::drawRunText(QImage& img, const RunTextConfig& config)
		{
			QVector<QString> textList;
			std::vector<rw::rqw::ImagePainter::PainterConfig> configList;
			rw::rqw::ImagePainter::PainterConfig painterConfig;
			painterConfig.textColor = rqw::RQWColorToQColor(config.operatorTimeTextColor);
			if (config.isDisOperatorTime)
			{
				configList.push_back(painterConfig);
				textList.push_back(config.operatorTimeText);
			}
			painterConfig.textColor =rqw::RQWColorToQColor(config.processImgTimeTextColor);
			if (config.isDisProcessImgTime)
			{
				configList.push_back(painterConfig);
				textList.push_back(config.processImgTimeText);
			}


			if (config.isDrawExtraText)
			{
				painterConfig.textColor = rqw::RQWColorToQColor(config.extraTextColor);
				configList.push_back(painterConfig);
				for (const auto& item : config.extraTexts)
				{
					textList.push_back(item);
				}
			}

			rw::rqw::ImagePainter::drawTextOnImage(img, textList, configList);
		}
	}
}
