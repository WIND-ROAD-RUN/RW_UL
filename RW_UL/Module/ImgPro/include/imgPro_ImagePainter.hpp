#pragma once

#include <QImage>

#include "ime_utilty.hpp"
#include "imgPro_ImageProcessUtilty.hpp"
#include"rqw_rqwColor.hpp"

namespace rw
{
	namespace imgPro
	{
		struct ConfigDrawLine
		{
			int position = 0;
			int thickness = 1;
			Color color = Color::Red;
		};

		struct ConfigDrawRect
		{
		public:
			int thickness = 1;
			Color rectColor = Color::Red;
			QString text;
			Color textColor= Color::Red;
			int fontSize = 3;
		public:
			enum class TextLocate
			{
				LeftTopIn,
				LeftTopOut,
				RightTopIn,
				RightTopOut,
				LeftBottomIn,
				LeftBottomOut,
				RightBottomIn,
				RightBottomOut,
				CenterIn,
			};
			TextLocate textLocate= TextLocate::LeftTopOut;
		};

		struct ImagePainter
		{
		public:
			static void drawVerticalLine(QImage& image, const ConfigDrawLine & cfg);
			static void drawHorizontalLine(QImage& image, const ConfigDrawLine& cfg);

			static void drawShapesOnSourceImg(QImage& image, const DetectionRectangleInfo& rectInfo, const ConfigDrawRect & cfg);
			
			static void drawTextOnImage(QImage& image, const QVector<QString>& texts, const std::vector<Color>& colorList, double proportion);
			
		};
	}
}
