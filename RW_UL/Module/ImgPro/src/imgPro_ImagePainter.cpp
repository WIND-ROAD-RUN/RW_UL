#include"imgPro_ImagePainter.hpp"

#include <QPainter>

namespace rw
{
	namespace imgPro
	{
		void ImagePainter::drawVerticalLine(QImage& image, const ConfigDrawLine& cfg)
		{
			QPainter painter(&image);
			painter.setPen(QPen(rw::rqw::RQWColorToQColor(cfg.color), cfg.thickness));
			painter.drawLine(cfg.position, 0, cfg.position, image.height());
		}

		void ImagePainter::drawHorizontalLine(QImage& image, const ConfigDrawLine& cfg)
		{
			QPainter painter(&image);
			painter.setPen(QPen(rw::rqw::RQWColorToQColor(cfg.color), cfg.thickness));
			painter.drawLine(0, cfg.position, image.width(), cfg.position);
		}

		void ImagePainter::drawShapesOnSourceImg(QImage& image, const DetectionRectangleInfo& rectInfo,
			const ConfigDrawRect& cfg)
		{
			QPainter painter(&image);
			painter.setPen(QPen(rw::rqw::RQWColorToQColor(cfg.rectColor), cfg.thickness));

			QPolygonF obbPolygon;
			obbPolygon << QPointF(rectInfo.leftTop.first, rectInfo.leftTop.second)
				<< QPointF(rectInfo.rightTop.first, rectInfo.rightTop.second)
				<< QPointF(rectInfo.rightBottom.first, rectInfo.rightBottom.second)
				<< QPointF(rectInfo.leftBottom.first, rectInfo.leftBottom.second);

			painter.drawPolygon(obbPolygon);

			// 直接用同一个 painter 画文字
			painter.setPen(rw::rqw::RQWColorToQColor(cfg.textColor));
			QFont font = painter.font();
			font.setPixelSize(cfg.fontSize);
			painter.setFont(font);

			QPointF textPosition;
			int offset = cfg.fontSize;
			switch (cfg.textLocate) {
			case ConfigDrawRect::TextLocate::LeftTopIn:
				textPosition = QPointF(rectInfo.leftTop.first + 10, rectInfo.leftTop.second + offset);
				break;
			case ConfigDrawRect::TextLocate::LeftTopOut:
				textPosition = QPointF(rectInfo.leftTop.first, rectInfo.leftTop.second - 10);
				break;
			case ConfigDrawRect::TextLocate::RightTopIn:
				textPosition = QPointF(rectInfo.rightTop.first - offset * cfg.text.size() / 1.5, rectInfo.rightTop.second + offset);
				break;
			case ConfigDrawRect::TextLocate::RightTopOut:
				textPosition = QPointF(rectInfo.rightTop.first - offset * cfg.text.size() / 1.5, rectInfo.rightTop.second - 10);
				break;
			case ConfigDrawRect::TextLocate::LeftBottomIn:
				textPosition = QPointF(rectInfo.leftBottom.first + 10, rectInfo.leftBottom.second - 10);
				break;
			case ConfigDrawRect::TextLocate::LeftBottomOut:
				textPosition = QPointF(rectInfo.leftBottom.first, rectInfo.leftBottom.second + offset);
				break;
			case ConfigDrawRect::TextLocate::RightBottomIn:
				textPosition = QPointF(rectInfo.rightBottom.first - offset * cfg.text.size() / 1.5, rectInfo.rightBottom.second - 10);
				break;
			case ConfigDrawRect::TextLocate::RightBottomOut:
				textPosition = QPointF(rectInfo.rightBottom.first - offset * cfg.text.size() / 1.5, rectInfo.rightBottom.second + offset);
				break;
			case ConfigDrawRect::TextLocate::CenterIn:
				textPosition = QPointF(rectInfo.center_x, rectInfo.center_y);
				break;
			default:
				throw std::invalid_argument("Unsupported TextLocate type.");
			}

			painter.drawText(textPosition, cfg.text);
		}

		void ImagePainter::drawTextOnImage(QImage& image, const QVector<QString>& texts,
			const std::vector<Color>& colorList, double proportion)
		{
			if (texts.empty() || proportion <= 0.0 || proportion > 1.0) {
				return;
			}

			QPainter painter(&image);
			painter.setRenderHint(QPainter::Antialiasing);

			//calculate the font size based on the image height and the proportion
			//getting the image height
			int imageHeight = image.height();
			int fontSize = static_cast<int>(imageHeight * proportion);

			QFont font = painter.font();
			font.setPixelSize(fontSize);
			painter.setFont(font);

			int x = 0;
			int y = fontSize;

			for (size_t i = 0; i < texts.size(); ++i) {
				QColor color = (i < colorList.size()) ? rw::rqw::RQWColorToQColor(colorList[i]) : rw::rqw::RQWColorToQColor(colorList.back());
				painter.setPen(color);

				painter.drawText(x, y, texts[i]);

				y += fontSize;
			}

			painter.end();
		}
	}
}