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

			rw::imgPro::ConfigDrawRect painterConfig;
			painterConfig.fontSize = config.fontSize;
			painterConfig.thickness = config.thickness;
			painterConfig.textLocate = config.textLocate;
			if (config.isDrawDefects)
			{
				painterConfig.rectColor = Color::Red;
				painterConfig.textColor = Color::Red;
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
							painterConfig.rectColor = findColor->second;
							painterConfig.textColor = findColor->second;
						}
						
						rw::imgPro::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
					}
				}
			}

			if (config.isDrawDisableDefects)
			{
				painterConfig.rectColor = Color::Green;
				painterConfig.textColor = Color::Green;

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
							painterConfig.rectColor = findColor->second;
							painterConfig.textColor = findColor->second;
						}

						rw::imgPro::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
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
			textColor =config.processImgTimeTextColor;
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

			rw::imgPro::ImagePainter::drawTextOnImage(img, textList, configList,config.runTextProportion);
		}
	}
}
