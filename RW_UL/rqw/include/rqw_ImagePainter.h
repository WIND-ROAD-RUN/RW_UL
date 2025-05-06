#pragma once

#include<QImage>

#include"ime_utilty.hpp"

namespace rw {
	namespace rqw {
		struct ImagePainter
		{
			enum class ShapeType {
				Rectangle,
				Circle
			};

			enum class BasicColor {
				Red,
				Green,
				Blue,
				Yellow,
				Cyan,
				Magenta,
				White,
				Black,
				Orange,
				LightBlue,
				Gray,
				Purple,
				Brown,
				LightBrown
			};

			static QColor toQColor(BasicColor color);

			struct PainterConfig
			{
				ShapeType shapeType{ ShapeType::Rectangle };
				int fontSize = 25;
				int fontThickness = 1;
				int thickness = 2;
				QString text;
				QColor color{ Qt::red };
				QColor textColor{ Qt::green };
			};

			static void drawTextOnImage(QImage& image, const QVector<QString>& texts, const std::vector<PainterConfig>& colorList, double proportion = 0.08);

			static QImage drawShapes(
				const QImage& image,
				const std::vector<DetectionRectangleInfo>& rectInfo,
				PainterConfig config
			);

			static void drawShapesOnSourceImg(
				QImage& image,
				const std::vector<DetectionRectangleInfo>& rectInfo,
				PainterConfig config = {}
			);

			static void drawShapesOnSourceImg(QImage& image, const std::vector<std::vector<size_t>> index, const std::vector<DetectionRectangleInfo>& rectInfo,
				PainterConfig config = {});

			static QImage drawShapes(
				const QImage& image,
				const DetectionRectangleInfo& rectInfo,
				PainterConfig config = {}
			);

			static void drawShapesOnSourceImg(
				QImage& image,
				const DetectionRectangleInfo& rectInfo,
				PainterConfig config = {}
			);

			static void drawVerticalLine(QImage& image, int position, const ImagePainter::PainterConfig& config);
			static void drawHorizontalLine(QImage& image, int position, const ImagePainter::PainterConfig& config);
		};
	
	}
}