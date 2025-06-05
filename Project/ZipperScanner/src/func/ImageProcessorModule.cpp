#include "ImageProcessorModule.h"

#include <qcolor.h>
#include <QPainter>

#include "GlobalStruct.hpp"
#include"rqw_ImagePainter.h"


QColor ImagePainter::ColorToQColor(Color c)
{
	switch (c) {
	case Color::White:   return QColor(255, 255, 255);
	case Color::Red:     return QColor(255, 0, 0);
	case Color::Green:   return QColor(0, 255, 0);
	case Color::Blue:    return QColor(0, 0, 255);
	case Color::Yellow:  return QColor(255, 255, 0);
	case Color::Cyan:    return QColor(0, 255, 255);
	case Color::Magenta: return QColor(255, 0, 255);
	case Color::Black:   return QColor(0, 0, 0);
	default:             return QColor(255, 255, 255);
	}
}

void ImagePainter::drawTextOnImage(QImage& image, const QVector<QString>& texts, const QVector<Color>& colorList, double proportion)
{
	if (texts.isEmpty() || proportion <= 0.0 || proportion > 1.0) {
		return; // 无效输入直接返回
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
	int y = 0;

	// 绘制每一行文字
	for (int i = 0; i < texts.size(); ++i) {
		// 获取颜色
		QColor color = (i < colorList.size()) ? ColorToQColor(colorList[i]) : ColorToQColor(colorList.last());
		painter.setPen(color);

		// 绘制文字
		painter.drawText(x, y + fontSize, texts[i]);

		// 更新 y 坐标
		y += fontSize; // 每行文字的间距等于字体大小
	}

	painter.end();
}

void ImageProcessor::drawZipperDefectInfoText_defect(QImage& image, const ZipperDefectInfo& info)
{
	QVector<QString> textList;
	std::vector<rw::rqw::ImagePainter::PainterConfig> configList;
	rw::rqw::ImagePainter::PainterConfig config;

	// 添加绿色与红色
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
	configList.push_back(config);
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
	configList.push_back(config);

	//运行时间
	textList.push_back(info.time);

	auto& generalSet = GlobalStructDataZipper::getInstance().generalConfig;
	auto& isDefect = generalSet.isDefect;

	// 如果开启了剔废功能
	if (isDefect)
	{
		// 添加剔废信息(如果信息内容太多记得修改)
		appendQueyaDectInfo(textList, info);
		appendTangshangDectInfo(textList, info);
		appendZangwuDectInfo(textList, info);
	}

	// 将信息显示到左上角
	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList);
}

void ImageProcessor::appendQueyaDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info)
{
	auto& productScore = GlobalStructDataZipper::getInstance().scoreConfig;
	if (_isbad && productScore.queYa && !info.queYaList.empty())
	{
		QString queyaText("缺牙:");
		for (const auto& item : info.queYaList)
		{
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area,0,'f',2));
		}
		queyaText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.queYaScore)).arg(static_cast<int>(productScore.queYaArea)));
		textList.push_back(queyaText);
	}
}

void ImageProcessor::appendTangshangDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info)
{
	auto& productScore = GlobalStructDataZipper::getInstance().scoreConfig;
	if (_isbad && productScore.tangShang && !info.tangShangList.empty())
	{
		QString tangshangText("烫伤:");
		for (const auto& item : info.tangShangList)
		{
			tangshangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		tangshangText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.tangShangScore)).arg(static_cast<int>(productScore.tangShangArea)));
		textList.push_back(tangshangText);
	}
}

void ImageProcessor::appendZangwuDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info)
{
	auto& productScore = GlobalStructDataZipper::getInstance().scoreConfig;
	if (_isbad && productScore.zangWu && !info.zangWuList.empty())
	{
		QString zangwuText("脏污:");
		for (const auto& item : info.zangWuList)
		{
			zangwuText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		zangwuText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.zangWuScore)).arg(static_cast<int>(productScore.zangWuArea)));
		textList.push_back(zangwuText);
	}
}


void ImageProcessingModule::onFrameCaptured(cv::Mat frame,size_t index)
{
	emit imgForDlgNewProduction(frame, index);

	if (frame.empty()) {
		return; // 跳过空帧
	}

	QMutexLocker locker(&_mutex);
	MatInfo mat;
	mat.image = frame;
	mat.index = index;
	_queue.enqueue(mat);
	_condition.wakeOne();
}
