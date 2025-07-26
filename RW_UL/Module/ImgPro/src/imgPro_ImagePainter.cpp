#include"imgPro_ImagePainter.hpp"

#include <QPainter>

#include "rqw_ImgConvert.hpp"

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

			if (cfg.isRegion)
			{

				painter.setRenderHint(QPainter::Antialiasing);
				painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

				// 构造多边形区域
				QPolygonF regionPolygon;
				regionPolygon << QPointF(rectInfo.leftTop.first, rectInfo.leftTop.second)
					<< QPointF(rectInfo.rightTop.first, rectInfo.rightTop.second)
					<< QPointF(rectInfo.rightBottom.first, rectInfo.rightBottom.second)
					<< QPointF(rectInfo.leftBottom.first, rectInfo.leftBottom.second);

				QColor fillColor = rw::rqw::RQWColorToQColor(cfg.rectColor);
				fillColor.setAlphaF(cfg.alpha); // alpha为0~1

				painter.setPen(Qt::NoPen);
				painter.setBrush(QBrush(fillColor));
				painter.drawPolygon(regionPolygon);

				if (cfg.hasFrame)
				{
					painter.setPen(QPen(rw::rqw::RQWColorToQColor(cfg.rectColor), cfg.thickness));
					painter.setBrush(Qt::NoBrush);
					painter.drawPolygon(regionPolygon);
				}
			}
			else
			{
				painter.setPen(QPen(rw::rqw::RQWColorToQColor(cfg.rectColor), cfg.thickness));

				QPolygonF obbPolygon;
				obbPolygon << QPointF(rectInfo.leftTop.first, rectInfo.leftTop.second)
					<< QPointF(rectInfo.rightTop.first, rectInfo.rightTop.second)
					<< QPointF(rectInfo.rightBottom.first, rectInfo.rightBottom.second)
					<< QPointF(rectInfo.leftBottom.first, rectInfo.leftBottom.second);

				painter.drawPolygon(obbPolygon);

			}

			if (cfg.text.isEmpty())
			{
				return;
			}

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

		void ImagePainter::drawTextOnImageWithFontSize(QImage& image, const QVector<QString>& texts,
			const std::vector<Color>& colorList, int fontSize)
		{
			if (texts.empty() || fontSize <= 0) {
				return;
			}

			QPainter painter(&image);
			painter.setRenderHint(QPainter::Antialiasing);

			QFont font = painter.font();
			font.setPixelSize(fontSize);
			painter.setFont(font);

			int x = 0;
			int y = fontSize;

			for (int i = 0; i < texts.size(); ++i) {
				QColor color = (i < static_cast<int>(colorList.size())) ? rw::rqw::RQWColorToQColor(colorList[i]) : rw::rqw::RQWColorToQColor(colorList.back());
				painter.setPen(color);

				painter.drawText(x, y, texts[i]);

				y += fontSize;
			}

			painter.end();
		}

		void ImagePainter::drawMaskOnSourceImg(QImage& image, const DetectionRectangleInfo& rectInfo,
			const ConfigDrawMask& cfg)
		{
			// 1. 检查 mask 是否为空
			if (rectInfo.mask_roi.empty()) {
				return;
			}

			// 2. 获取 ROI 区域
			QRect roi(rectInfo.roi.x, rectInfo.roi.y, rectInfo.roi.width, rectInfo.roi.height);
			if (!image.rect().contains(roi)) {
				return;
			}

			// 3. 将 mask_roi 转为 QImage
			QImage maskImg = rw::CvMatToQImage(rectInfo.mask_roi);

			// 4. 阈值处理
			for (int y = 0; y < maskImg.height(); ++y) {
				uchar* line = maskImg.scanLine(y);
				for (int x = 0; x < maskImg.width(); ++x) {
					line[x] = (line[x] > cfg.thresh) ? static_cast<uchar>(cfg.maxVal) : 0;
				}
			}

			// 5. 构造彩色遮罩
			QColor color = rw::rqw::RQWColorToQColor(cfg.maskColor);
			QImage colorMask(maskImg.size(), QImage::Format_ARGB32);
			colorMask.fill(Qt::transparent);
			for (int y = 0; y < maskImg.height(); ++y) {
				const uchar* maskLine = maskImg.constScanLine(y);
				QRgb* colorLine = reinterpret_cast<QRgb*>(colorMask.scanLine(y));
				for (int x = 0; x < maskImg.width(); ++x) {
					if (maskLine[x]) {
						colorLine[x] = QColor(color.red(), color.green(), color.blue(), static_cast<int>(cfg.alpha * 255)).rgba();
					}
				}
			}

			// 6. 叠加到原图 ROI 区域
			QPainter painter(&image);
			painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
			painter.drawImage(roi.topLeft(), colorMask);

			painter.end();

			if (cfg.hasFrame)
			{
				drawShapesOnSourceImg(image, rectInfo, cfg.rectCfg);
			}


		}
	}
}
