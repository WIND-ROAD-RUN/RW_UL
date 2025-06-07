#include "ImageProcessorModule.h"

#include <qcolor.h>
#include <QPainter>

#include "GlobalStruct.hpp"
#include"rqw_ImagePainter.h"
#include "Utilty.hpp"


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

ImageProcessorZipper::ImageProcessorZipper(QQueue<MatInfo>& queue, QMutex& mutex, QWaitCondition& condition, int workIndex, QObject* parent)
	: QThread(parent), _queue(queue), _mutex(mutex), _condition(condition), _workIndex(workIndex){

}

void ImageProcessorZipper::run()
{
	while (!QThread::currentThread()->isInterruptionRequested()) {
		MatInfo frame;
		{
			QMutexLocker locker(&_mutex);
			if (_queue.isEmpty()) {
				_condition.wait(&_mutex);
				if (QThread::currentThread()->isInterruptionRequested()) {
					break;
				}
			}
			if (!_queue.isEmpty()) {
				frame = _queue.dequeue();
			}
			else {
				continue; // 如果队列仍为空，跳过本次循环
			}
		}

		// 检查 frame 是否有效
		if (frame.image.empty()) {
			continue; // 跳过空帧
		}

		auto& globalData = GlobalStructDataZipper::getInstance();

		auto currentRunningState = globalData.runningState.load();
		switch (currentRunningState)
		{
		case RunningState::Debug:
			run_debug(frame);
			break;
		case RunningState::OpenRemoveFunc:
			run_OpenRemoveFunc(frame);
			break;
		case RunningState::Monitor:
			run_monitor(frame);
			break;
		default:
			break;
		}
	}

}

void ImageProcessorZipper::run_debug(MatInfo& frame)
{

}

void ImageProcessorZipper::run_monitor(MatInfo& frame)
{

}

void ImageProcessorZipper::run_OpenRemoveFunc(MatInfo& frame)
{

}

void ImageProcessorZipper::run_OpenRemoveFunc_process_defect_info(ZipperDefectInfo& info)
{
	_isbad = false; // 重置坏品标志
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& isOpenDefect = globalStruct.generalConfig.isDefect;

	if (isOpenDefect)
	{
		run_OpenRemoveFunc_process_defect_info_QueYa(info);
		run_OpenRemoveFunc_process_defect_info_TangShang(info);
		run_OpenRemoveFunc_process_defect_info_ZangWu(info);
	}
}

void ImageProcessorZipper::run_OpenRemoveFunc_process_defect_info_QueYa(ZipperDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.queYa)
	{
		auto& queya = info.queYaList;
		if (queya.empty())
		{
			return;
		}
		for (const auto& item : queya)
		{
			if (item.score >= productScore.queYaScore)
			{
				_isbad = true; // 有缺牙就认为是坏品
				break; // 找到一个符合条件的缺牙就可以了
			}
		}
	}
}

void ImageProcessorZipper::run_OpenRemoveFunc_process_defect_info_TangShang(ZipperDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.tangShang)
	{
		auto& tangshang = info.tangShangList;
		if (tangshang.empty())
		{
			return;
		}
		for (const auto& item : tangshang)
		{
			if (item.score >= productScore.tangShangScore)
			{
				_isbad = true; // 有烫伤就认为是坏品
				break; // 找到一个符合条件的烫伤就可以了
			}
		}
	}
}

void ImageProcessorZipper::run_OpenRemoveFunc_process_defect_info_ZangWu(ZipperDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& productScore = globalStruct.scoreConfig;

	if (productScore.zangWu)
	{
		auto& Zangwu = info.zangWuList;
		if (Zangwu.empty())
		{
			return;
		}
		for (const auto& item : Zangwu)
		{
			if (item.score >= productScore.zangWuScore)
			{
				_isbad = true; // 有脏污就认为是坏品
				break; // 找到一个符合条件的脏污就可以了
			}
		}
	}

}

void ImageProcessorZipper::run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const
{

}

void ImageProcessorZipper::getEliminationInfo_debug(ZipperDefectInfo& info,
                                              const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index,
                                              const cv::Mat& mat)
{
	getQueyaInfo(info, processResult, index[ClassId::Queya]);
	getTangshangInfo(info, processResult, index[ClassId::Tangshang]);
	getZangwuInfo(info, processResult, index[ClassId::Zangwu]);
}

void ImageProcessorZipper::getEliminationInfo_defect(ZipperDefectInfo& info,
                                               const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index,
                                               const cv::Mat& mat)
{
	getQueyaInfo(info, processResult, index[ClassId::Queya]);
	getTangshangInfo(info, processResult, index[ClassId::Tangshang]);
	getZangwuInfo(info, processResult, index[ClassId::Zangwu]);
}

void ImageProcessorZipper::getQueyaInfo(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult,
                                  const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto queyaScore = processResult[item].score * 100; // 将分数转换为百分比
		auto queyaArea = static_cast<double>(processResult[item].area); // 获取面积
		info.queYaList.emplace_back(ZipperDefectInfo::DetectItem{ queyaScore, queyaArea, static_cast<int>(item) });
	}
}

void ImageProcessorZipper::getTangshangInfo(ZipperDefectInfo& info,
                                      const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto tangshangScore = processResult[item].score * 100; // 将分数转换为百分比
		auto tangshangArea = static_cast<double>(processResult[item].area); // 获取面积
		info.tangShangList.emplace_back(ZipperDefectInfo::DetectItem{ tangshangScore, tangshangArea, static_cast<int>(item) });
	}
}

void ImageProcessorZipper::getZangwuInfo(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult,
                                   const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}

	for (const auto& item : processIndex)
	{
		auto zangwuScore = processResult[item].score * 100; // 将分数转换为百分比
		auto zangwuArea = static_cast<double>(processResult[item].area); // 获取面积

		info.zangWuList.emplace_back(ZipperDefectInfo::DetectItem{ zangwuScore, zangwuArea, static_cast<int>(item) });

	}
}

std::vector<std::vector<size_t>> ImageProcessorZipper::getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info)
{
	std::vector<std::vector<size_t>> result;
	result.resize(20);

	for (int i = 0;i < info.size();i++)
	{
		if (info[i].classId > result.size())
		{
			result.resize(info[i].classId + 1);
		}

		result[info[i].classId].emplace_back(i);
	}

	return result;
}

void ImageProcessorZipper::buildSegModelEngine(const QString& enginePath)
{
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.1f;
	config.nms_threshold = 0.1f;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::LetterBox;
	config.letterBoxColor = cv::Scalar(114, 114, 114);
	config.modelPath = enginePath.toStdString();
	_modelEngine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::Yolov11_Det, rw::ModelEngineDeployType::TensorRT);
}

std::vector<std::vector<size_t>> ImageProcessorZipper::filterEffectiveIndexes_debug(std::vector<rw::DetectionRectangleInfo> info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	auto processIndex = getClassIndex(info);
	processIndex = getIndexInBoundary(info, processIndex);
	return processIndex;
}

std::vector<std::vector<size_t>> ImageProcessorZipper::filterEffectiveIndexes_defect(std::vector<rw::DetectionRectangleInfo> info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	auto processIndex = getClassIndex(info);
	processIndex = getIndexInBoundary(info, processIndex);

	return processIndex;
}

std::vector<std::vector<size_t>> ImageProcessorZipper::getIndexInBoundary(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index)
{
	std::vector<std::vector<size_t>> result;
	result.resize(index.size());
	for (size_t i = 0; i < index.size(); i++)
	{
		for (size_t j = 0; j < index[i].size(); j++)
		{
			if (isInBoundary(info[index[i][j]]))
			{
				result[i].push_back(index[i][j]);
			}
		}
	}
	return result;
}


bool ImageProcessorZipper::isInBoundary(const rw::DetectionRectangleInfo& info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto x = info.center_x;
	auto y = info.center_y;

	if (imageProcessingModuleIndex == 1)
	{
		auto lineLeft = globalStruct.setConfig.zuoXianWei1;
		auto lineRight = globalStruct.setConfig.youXianWei1;
		auto lineTop = globalStruct.setConfig.shangXianWei1;
		auto lineBottom = globalStruct.setConfig.xiaXianWei1;
		if (lineLeft < x && x < lineRight)
		{
			if (lineTop < y && y < lineBottom)
			{
				return true;
			}
		}
	}
	else if (imageProcessingModuleIndex == 2)
	{
		auto lineLeft = globalStruct.setConfig.zuoXianWei2;
		auto lineRight = globalStruct.setConfig.youXianWei2;
		auto lineTop = globalStruct.setConfig.shangXianWei2;
		auto lineBottom = globalStruct.setConfig.xiaXianWei2;
		if (lineLeft < x && x < lineRight)
		{
			if (lineTop < y && y < lineBottom)
			{
				return true;
			}
		}
	}

	return false;
}

void ImageProcessorZipper::drawZipperDefectInfoText_defect(QImage& image, const ZipperDefectInfo& info)
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

void ImageProcessorZipper::appendQueyaDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info)
{
	auto& productScore = GlobalStructDataZipper::getInstance().scoreConfig;
	if (_isbad && productScore.queYa && !info.queYaList.empty())
	{
		QString queyaText("缺牙:");
		for (const auto& item : info.queYaList)
		{
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		queyaText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.queYaScore)).arg(static_cast<int>(productScore.queYaArea)));
		textList.push_back(queyaText);
	}
}

void ImageProcessorZipper::appendTangshangDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info)
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

void ImageProcessorZipper::appendZangwuDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info)
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

void ImageProcessorZipper::drawVerticalLine_locate(QImage& image, size_t locate)
{
	if (image.isNull() || locate >= static_cast<size_t>(image.width())) {
		return; // 如果图像无效或 locate 超出图像宽度，直接返回
	}

	QPainter painter(&image);
	painter.setRenderHint(QPainter::Antialiasing); // 开启抗锯齿
	painter.setPen(QPen(Qt::red, 2)); // 设置画笔颜色为红色，线宽为2像素

	// 绘制竖线，从图像顶部到底部
	painter.drawLine(QPoint(locate, 0), QPoint(locate, image.height()));

	painter.end(); // 结束绘制
}

void ImageProcessorZipper::drawHorizontalLine_locate(QImage& image, size_t locate)
{
	if (image.isNull() || locate >= static_cast<size_t>(image.height())) {
		return; // 如果图像无效或 locate 超出图像高度，直接返回
	}

	QPainter painter(&image);
	painter.setRenderHint(QPainter::Antialiasing); // 开启抗锯齿
	painter.setPen(QPen(Qt::red, 2)); // 设置画笔颜色为红色，线宽为2像素

	// 绘制横线，从图像左侧到右侧
	painter.drawLine(QPoint(0, locate), QPoint(image.width(), locate));

	painter.end(); // 结束绘制
}

void ImageProcessorZipper::drawZipperDefectInfoText_Debug(QImage& image, const ZipperDefectInfo& info)
{
	QVector<QString> textList;
	std::vector<rw::rqw::ImagePainter::PainterConfig> configList;
	rw::rqw::ImagePainter::PainterConfig config;

	configList.push_back(config);
	//运行时间
	textList.push_back(info.time);

	// 缺牙
	if (!info.queYaList.empty()) {
		QString queyaText("缺牙:");
		for (const auto& item : info.queYaList) {
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(queyaText);
	}
	// 烫伤
	if (!info.tangShangList.empty()) {
		QString tangshangText("烫伤:");
		for (const auto& item : info.tangShangList) {
			tangshangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(tangshangText);
	}
	// 脏污
	if (!info.zangWuList.empty()) {
		QString zangwuText("脏污:");
		for (const auto& item : info.zangWuList) {
			zangwuText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(zangwuText);
	}

	// 显示到左上角
	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList, 0.05);
}

void ImageProcessorZipper::drawDefectRec(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult,
	const std::vector<std::vector<size_t>>& processIndex)
{
	if (processResult.size() == 0)
	{
		return;
	}

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
	for (size_t i = 0; i < processIndex.size(); i++)
	{
		for (size_t j = 0; j < processIndex[i].size(); j++)
		{
			auto& item = processResult[processIndex[i][j]];
			switch (item.classId)
			{
			case ClassId::Queya:
				config.text = "缺牙 " + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
				break;
			case ClassId::Tangshang:
				config.text = "烫伤 " + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
				break;
			case ClassId::Zangwu:
				config.text = "脏污 " + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
				break;
			default:
				config.text = QString::number(item.classId) + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
				break;
			}
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
		}
	}
}

void ImageProcessorZipper::drawDefectRec_error(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult,
	const std::vector<std::vector<size_t>>& processIndex, const ZipperDefectInfo& info)
{
	auto& scoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	auto& generalConfig = GlobalStructDataZipper::getInstance().generalConfig;
	if (processResult.size() == 0)
	{
		return;
	}

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);

	for (size_t i = 0; i < processIndex.size(); i++)
	{
		for (size_t j = 0; j < processIndex.size(); j++)
		{
			auto& item = processResult[processIndex[i][j]];
			switch (item.classId)
			{
			case ClassId::Queya:
				if (generalConfig.isDefect && scoreConfig.queYa)
				{
					config.text = "缺牙 " + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
					break;
				}
			case ClassId::Tangshang:
				if (generalConfig.isDefect && scoreConfig.tangShang)
				{
					config.text = "烫伤 " + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
					break;
				}
			case ClassId::Zangwu:
				if (generalConfig.isDefect && scoreConfig.zangWu)
				{
					config.text = "脏污 " + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
					break;
				}
			default:
				config.text = QString::number(item.classId) + QString::number(qRound(item.score * 100)) + " " + QString::number(item.area);
				break;
			}

			// 如果开启了剔废
			if (generalConfig.isDefect)
			{
				if (item.classId == ClassId::Queya && scoreConfig.queYa)
				{
					for (const auto& detectItem : info.queYaList)
					{
						if (detectItem.index == processIndex[i][j])
						{
							rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
							break; // 找到匹配的就跳出循环
						}
					}
				}
				else if (item.classId == ClassId::Tangshang && scoreConfig.tangShang)
				{
					for (const auto& detectItem : info.tangShangList)
					{
						if (detectItem.index == processIndex[i][j])
						{
							rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
							break; // 找到匹配的就跳出循环
						}
					}
				}
				else if (item.classId == ClassId::Zangwu && scoreConfig.zangWu)
				{
					for (const auto& detectItem : info.zangWuList)
					{
						if (detectItem.index == processIndex[i][j])
						{
							rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
							break; // 找到匹配的就跳出循环
						}
					}
				}
			}
		}
	}
}


void ImageProcessingModuleZipper::onFrameCaptured(cv::Mat frame, size_t index)
{
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

void ImageProcessingModuleZipper::BuildModule()
{
	for (int i = 0; i < _numConsumers; ++i) {
		static size_t workIndexCount = 0;
		ImageProcessorZipper* processor = new ImageProcessorZipper(_queue, _mutex, _condition, workIndexCount, this);
		workIndexCount++;
		processor->buildSegModelEngine(modelEnginePath);
		processor->imageProcessingModuleIndex = index;
		connect(processor, &ImageProcessorZipper::imageReady, this, &ImageProcessingModuleZipper::imageReady, Qt::QueuedConnection);
		_processors.push_back(processor);
		processor->start();
	}
}

ImageProcessingModuleZipper::ImageProcessingModuleZipper(int numConsumers, QObject* parent)
	: QObject(parent), _numConsumers(numConsumers){

}

ImageProcessingModuleZipper::~ImageProcessingModuleZipper()
{
	// 通知所有线程退出
	for (auto processor : _processors) {
		processor->requestInterruption();
	}

	// 唤醒所有等待的线程
	{
		QMutexLocker locker(&_mutex);
		_condition.wakeAll();
	}

	// 等待所有线程退出
	for (auto processor : _processors) {
		if (processor->isRunning()) {
			processor->wait(1000); // 使用超时机制，等待1秒
		}
		delete processor;
	}
}
