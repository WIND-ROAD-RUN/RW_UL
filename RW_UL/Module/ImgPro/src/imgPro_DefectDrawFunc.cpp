#include"imgPro_DefectDrawFunc.hpp"

#include "imgPro_DefectResultInfoFunc.hpp"
#include "rqw_ImagePainter.h"

namespace rw
{
	namespace imgPro
	{
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
						rw::rqw::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
					}
				}
			}

		}
	}
}
