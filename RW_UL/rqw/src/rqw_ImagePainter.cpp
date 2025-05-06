#include "rqw_ImagePainter.h"

#include<QPainter>

namespace rw
{
	namespace rqw
	{
		QColor ImagePainter::toQColor(BasicColor color)
		{
			switch (color)
			{
			case BasicColor::Red:
				return QColor(Qt::red);
			case BasicColor::Green:
				return QColor(Qt::green);
			case BasicColor::Blue:
				return QColor(Qt::blue);
			case BasicColor::Yellow:
				return QColor(Qt::yellow);
			case BasicColor::Cyan:
				return QColor(Qt::cyan);
			case BasicColor::Magenta:
				return QColor(Qt::magenta);
			case BasicColor::White:
				return QColor(Qt::white);
			case BasicColor::Black:
				return QColor(Qt::black);
			case BasicColor::Orange:
				return QColor(255, 165, 0); // RGB for orange
			case BasicColor::LightBlue:
				return QColor(173, 216, 230); // RGB for light blue
			case BasicColor::Gray:
				return QColor(Qt::gray);
			case BasicColor::Purple:
				return QColor(128, 0, 128); // RGB for purple
			case BasicColor::Brown:
				return QColor(165, 42, 42); // RGB for brown
			case BasicColor::LightBrown:
				return QColor(210, 180, 140); // RGB for light brown
			default:
				return QColor(Qt::black); // Default to black if unknown color
			}
		}

		void ImagePainter::drawTextOnImage(QImage& image, const QVector<QString>& texts,
			const std::vector<PainterConfig>& colorList, double proportion)
		{
			if (texts.empty() || proportion <= 0.0 || proportion > 1.0) {
				return; 
			}

			QPainter painter(&image);
			painter.setRenderHint(QPainter::Antialiasing);

			// 计算字体大小
			int imageHeight = image.height();
			int fontSize = static_cast<int>(imageHeight * proportion); // 字号由 proportion 决定

			QFont font = painter.font();
			font.setPixelSize(fontSize);
			painter.setFont(font);

			// 起始位置
			int x = 0;
			int y = fontSize; // 初始 y 坐标为字体大小，避免文字超出顶部

			// 绘制每一行文字
			for (size_t i = 0; i < texts.size(); ++i) {
				// 获取颜色
				QColor color = (i < colorList.size()) ? colorList[i].textColor : colorList.back().textColor;
				painter.setPen(color);

				// 绘制文字
				painter.drawText(x, y, texts[i]);

				// 更新 y 坐标
				y += fontSize; // 每行文字的间距等于字体大小
			}

			painter.end();
		}

		QImage ImagePainter::drawShapes(const QImage& image, const std::vector<DetectionRectangleInfo>& rectInfo,
		                                PainterConfig config)
		{
			QImage resultImage = image.copy();
			for (const auto& item : rectInfo)
			{
				drawShapesOnSourceImg(resultImage, item, config);
			}
			return resultImage;
		}

		void ImagePainter::drawShapesOnSourceImg(QImage& image, const std::vector<DetectionRectangleInfo>& rectInfo,
		                                         PainterConfig config)
		{
			for (const auto& item : rectInfo)
			{
				config.text = QString::number(item.classId);
				config.fontSize = 25;
				drawShapesOnSourceImg(image, item, config);
			}
		}

		void ImagePainter::drawShapesOnSourceImg(QImage& image, const std::vector<std::vector<size_t>> index,
		                                         const std::vector<DetectionRectangleInfo>& rectInfo, PainterConfig config)
		{
			for (const auto& classId : index) {
				for (const auto& item : classId)
				{
					config.text = QString::number(rectInfo[item].classId);
					config.fontSize = 25;
					drawShapesOnSourceImg(image, rectInfo[item], config);
				}
			}
		}

		QImage ImagePainter::drawShapes(const QImage& image, const DetectionRectangleInfo& rectInfo, PainterConfig config)
		{
			QImage resultImage = image.copy();
			drawShapesOnSourceImg(resultImage, rectInfo, config);
			return resultImage;
		}
		void ImagePainter::drawShapesOnSourceImg(QImage& image, const DetectionRectangleInfo& rectInfo, PainterConfig config)
		{
			if (config.shapeType == ShapeType::Rectangle) {
				QPainter painter(&image);
				painter.setPen(QPen(config.color, config.thickness));
				painter.drawRect(
					QRectF(
						rectInfo.leftTop.first,
						rectInfo.leftTop.second,
						rectInfo.rightBottom.first - rectInfo.leftTop.first,
						rectInfo.rightBottom.second - rectInfo.leftTop.second
					)
				);
			}
			else if (config.shapeType == ShapeType::Circle) {
				QPainter painter(&image);
				painter.setPen(QPen(config.color, config.thickness));
				int radius = std::min(rectInfo.width, rectInfo.height) / 2;
				painter.drawEllipse(
					QPointF(rectInfo.center_x, rectInfo.center_y),
					radius,
					radius
				);
			}
			QPainter textPainter(&image);
			textPainter.setPen(config.textColor);

			// 设置字体大小
			QFont font = textPainter.font();
			font.setPixelSize(config.fontSize); 
			textPainter.setFont(font);

			// 绘制文字
			textPainter.drawText(
				QPointF(rectInfo.leftTop.first, rectInfo.leftTop.second - 10),
				config.text
			);
		}
		void ImagePainter::drawVerticalLine(QImage& image, int position, const ImagePainter::PainterConfig& config)
		{
			QPainter painter(&image);
			painter.setPen(QPen(config.color, config.thickness));
			painter.drawLine(position, 0, position, image.height());
		}
		void ImagePainter::drawHorizontalLine(QImage& image, int position, const ImagePainter::PainterConfig& config)
		{
			QPainter painter(&image);
			painter.setPen(QPen(config.color, config.thickness));
			painter.drawLine(0, position, image.width(), position);
		}
	}
}
