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

		QVector3D ImagePainter::calculateRegionRGB(const QImage& image, const DetectionRectangleInfo& total,
			CropMode mode, const QVector<DetectionRectangleInfo>& excludeRegions, CropMode excludeMode)
		{
			QRect rect_total(
				QPoint(total.leftTop.first, total.leftTop.second),
				QSize(total.width, total.height) 
			);

			QVector<QRect> rect_exclude;
			for (const auto& item : excludeRegions)
			{
				QRect rect_excludeItem(
					QPoint(item.leftTop.first, item.leftTop.second),
					QSize(item.width, item.height)
				);
				rect_exclude.push_back(rect_excludeItem);
			}
			return calculateRegionRGB(image, rect_total, mode, rect_exclude, excludeMode);

		}

		QVector3D ImagePainter::calculateRegionRGB(const QImage& image, const QRect& rect, CropMode mode,
		                                           const QVector<QRect>& excludeRegions, CropMode excludeMode)
		{
			// 检查图像是否为空
			if (image.isNull()) {
				throw std::invalid_argument("Input image is empty.");
			}

			// 检查图像是否为 RGB 格式
			if (image.format() != QImage::Format_RGB32 && image.format() != QImage::Format_ARGB32) {
				throw std::invalid_argument("Input image must be a 3-channel (RGB) image.");
			}

			// 确保矩形在图像范围内
			QRect validRect = rect.intersected(image.rect());
			if (validRect.isEmpty()) {
				throw std::invalid_argument("The rectangle is outside the image bounds.");
			}

			// 创建掩码
			QImage mask(image.size(), QImage::Format_Grayscale8);
			mask.fill(0);

			// 根据模式标记主区域
			QPainter painter(&mask);
			painter.setBrush(Qt::white);
			if (mode == CropMode::Rectangle) {
				painter.drawRect(validRect);
			}
			else if (mode == CropMode::InscribedCircle) {
				int radius = std::min(validRect.width(), validRect.height()) / 2;
				QPoint center(validRect.center());
				painter.drawEllipse(center, radius, radius);
			}
			else {
				throw std::invalid_argument("Invalid crop mode.");
			}

			// 处理需要排除的区域
			painter.setBrush(Qt::black);
			for (const auto& excludeRect : excludeRegions) {
				QRect validExcludeRect = excludeRect.intersected(validRect);
				if (validExcludeRect.isEmpty()) {
					continue; // 跳过无效的排除区域
				}

				if (excludeMode == CropMode::Rectangle) {
					painter.drawRect(validExcludeRect);
				}
				else if (excludeMode == CropMode::InscribedCircle) {
					int radius = std::min(validExcludeRect.width(), validExcludeRect.height()) / 2;
					QPoint center(validExcludeRect.center());
					painter.drawEllipse(center, radius, radius);
				}
			}
			painter.end();

			// 计算平均 RGB 值
			double totalR = 0, totalG = 0, totalB = 0;
			int pixelCount = 0;

			for (int y = 0; y < image.height(); ++y) {
				for (int x = 0; x < image.width(); ++x) {
					if (qGray(mask.pixel(x, y)) > 0) { // 检查掩码是否有效
						QColor color(image.pixel(x, y));
						totalR += color.red();
						totalG += color.green();
						totalB += color.blue();
						++pixelCount;
					}
				}
			}

			if (pixelCount == 0) {
				throw std::runtime_error("No valid pixels in the specified region.");
			}

			// 返回平均 RGB 值
			return QVector3D(totalR / pixelCount, totalG / pixelCount, totalB / pixelCount);
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
