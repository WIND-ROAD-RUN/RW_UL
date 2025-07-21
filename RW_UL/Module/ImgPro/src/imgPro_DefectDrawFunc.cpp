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
				for (const auto& pairs : info.defects)
				{
					rw::rqw::ImagePainter::PainterConfig painterConfig;
					painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
					painterConfig.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
					painterConfig.text= config.classIdNameMap.find(pairs.first) ?
						config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
					for (const auto& item : pairs.second)
					{
						rw::rqw::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
					}
				}
			}

			if (config.isDrawDisableDefects)
			{
				for (const auto& pairs : info.disableDefects)
				{
					rw::rqw::ImagePainter::PainterConfig painterConfig;
					painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
					painterConfig.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
					painterConfig.text = config.classIdNameMap.find(pairs.first) ?
						config.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
					for (const auto& item : pairs.second)
					{
						rw::rqw::ImagePainter::drawShapesOnSourceImg(img, processResult[item.index], painterConfig);
					}
				}
			}

		}
	}
}
