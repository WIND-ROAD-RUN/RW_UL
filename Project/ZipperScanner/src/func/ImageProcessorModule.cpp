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
	: QThread(parent), _queue(queue), _mutex(mutex), _condition(condition), _workIndex(workIndex) {

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
			/*case RunningState::Monitor:
				run_monitor(frame);
				break;*/
		default:
			break;
		}
	}

}

void ImageProcessorZipper::run_debug(MatInfo& frame)
{
	//AI开始识别
	ZipperDefectInfo defectInfo;
	auto startTime = std::chrono::high_resolution_clock::now();

	auto processResult = _modelEngine->processImg(frame.image);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	defectInfo.time = QString("处理时间: %1 ms").arg(duration);
	//AI识别完成

	//过滤出有效索引
	auto processResultIndex = filterEffectiveIndexes_debug(processResult);
	//获取到当前图像的缺陷信息
	getEliminationInfo_debug(defectInfo, processResult, processResultIndex, frame.image);

	//绘制defect信息
	auto qImage = cvMatToQImage(frame.image);

	auto& generalConfig = GlobalStructDataZipper::getInstance().generalConfig;

	if (GlobalStructDataZipper::getInstance().debug_isDisplayRec)
	{
		drawBoundariesLines(qImage);


		// 如果勾选主窗体识别框才会绘制识别框
		if (generalConfig.isshibiekuang == true)
		{
			drawDefectRec(qImage, processResult, processResultIndex, defectInfo);

			drawDefectRec_error(qImage, processResult, processResultIndex, defectInfo);
		}
	}
	if (GlobalStructDataZipper::getInstance().debug_isDisplayText)
	{
		if (generalConfig.iswenzi == true)
		{
			drawZipperDefectInfoText_Debug(qImage, defectInfo);
		}
	}

	QPixmap pixmap = QPixmap::fromImage(qImage);

	emit imageReady(pixmap);
}

void ImageProcessorZipper::run_monitor(MatInfo& frame)
{

}

void ImageProcessorZipper::run_OpenRemoveFunc(MatInfo& frame)
{
	//AI开始识别
	ZipperDefectInfo defectInfo;
	auto startTime = std::chrono::high_resolution_clock::now();

	auto processResult = _modelEngine->processImg(frame.image);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	defectInfo.time = QString("处理时间: %1 ms").arg(duration);
	//AI识别完成

	//过滤出有效索引
	auto processResultIndex = filterEffectiveIndexes_defect(processResult);
	//获取到当前图像的缺陷信息
	getEliminationInfo_defect(defectInfo, processResult, processResultIndex, frame.image);

	// 剔除逻辑获取_isbad以及绘制defect错误信息
	run_OpenRemoveFunc_process_defect_info(defectInfo);
	//如果_isbad为true，将错误信息发送到剔除队列中
	run_OpenRemoveFunc_emitErrorInfo(frame);

	//绘制defect信息
	auto qImage = cvMatToQImage(frame.image);

	// 画限位线
	drawBoundariesLines(qImage);
	// 不满足剔废条件的缺陷用绿色显示
	drawDefectRec(qImage, processResult, processResultIndex, defectInfo);
	// 满足剔废条件的缺陷用红色显示
	drawDefectRec_error(qImage, processResult, processResultIndex, defectInfo);

	drawZipperDefectInfoText_defect(qImage, defectInfo);

	rw::rqw::ImageInfo imageInfo(cvMatToQImage(frame.image));
	//保存图像
	save_image(imageInfo, qImage);

	QPixmap pixmap = QPixmap::fromImage(qImage);

	// 显示图像
	emit imageReady(pixmap);

	// 显示NG图像
	emit imageNGReady(pixmap, imageProcessingModuleIndex,_isbad);
}

void ImageProcessorZipper::run_OpenRemoveFunc_process_defect_info(ZipperDefectInfo& info)
{
	_isbad = false; // 重置坏品标志
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	run_OpenRemoveFunc_process_defect_info_QueYa(info);
	run_OpenRemoveFunc_process_defect_info_TangShang(info);
	run_OpenRemoveFunc_process_defect_info_ZangWu(info);

}

void ImageProcessorZipper::run_OpenRemoveFunc_process_defect_info_QueYa(ZipperDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& productSet = globalStruct.scoreConfig;
	if (productSet.queYa)
	{
		auto& queya = info.queYaList;
		if (queya.empty())
		{
			return;
		}
		for (const auto& item : queya)
		{
			if (item.score >= productSet.queYaScore && item.area >= productSet.queYaArea)
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
	auto& productSet = globalStruct.scoreConfig;
	if (productSet.tangShang)
	{
		auto& tangshang = info.tangShangList;
		if (tangshang.empty())
		{
			return;
		}
		for (const auto& item : tangshang)
		{
			if (item.score >= productSet.tangShangScore && item.area >= productSet.tangShangArea)
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
	auto& productSet = globalStruct.scoreConfig;

	if (productSet.zangWu)
	{
		auto& Zangwu = info.zangWuList;
		if (Zangwu.empty())
		{
			return;
		}
		for (const auto& item : Zangwu)
		{
			if (item.score >= productSet.zangWuScore && item.area >= productSet.zangWuArea)
			{
				_isbad = true; // 有脏污就认为是坏品
				break; // 找到一个符合条件的脏污就可以了
			}
		}
	}
}

void ImageProcessorZipper::run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	if (_isbad)
	{
		++globalStruct.statisticalInfo.wasteCount;
	}

	if (imageProcessingModuleIndex == 1 || imageProcessingModuleIndex == 2)
	{
		++globalStruct.statisticalInfo.produceCount;
	}

	if (imageProcessingModuleIndex == 1)
	{
		++globalStruct.statisticalInfo.produceCount1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		++globalStruct.statisticalInfo.produceCount2;
	}

	if (_isbad)
	{
		switch (imageProcessingModuleIndex)
		{
		case 1:
			globalStruct.priorityQueue1->insert(frame.location, frame.location);
			break;
		case 2:
			globalStruct.priorityQueue2->insert(frame.location, frame.location);
			break;
		default:
			break;
		}
	}
}

void ImageProcessorZipper::save_image(rw::rqw::ImageInfo& imageInfo, const QImage& image)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& setConfig = globalStruct.setConfig;

	if (!globalStruct.isTakePictures)
	{
		return;
	}

	if (imageProcessingModuleIndex == 1 && setConfig.takeWork1Pictures)
	{
		if (globalStruct.isTakePictures) {
			save_image_work(imageInfo, image);
		}
	}
	else if (imageProcessingModuleIndex == 2 && setConfig.takeWork2Pictures)
	{
		if (globalStruct.isTakePictures) {
			save_image_work(imageInfo, image);
		}
	}
}

void ImageProcessorZipper::save_image_work(rw::rqw::ImageInfo& imageInfo, const QImage& image)
{
	auto& globalData = GlobalStructDataZipper::getInstance();
	auto& setConfig = globalData.setConfig;
	if (_isbad) {
		if (setConfig.saveNGImg)
		{
			imageInfo.classify = "NG";
			globalData.imageSaveEngine->pushImage(imageInfo);
		}
		if (setConfig.saveMaskImg)
		{
			rw::rqw::ImageInfo mask(image);
			mask.classify = "Mask";
			globalData.imageSaveEngine->pushImage(mask);
		}
	}
	else {
		if (setConfig.saveOKImg)
		{
			imageInfo.classify = "OK";
			globalData.imageSaveEngine->pushImage(imageInfo);
		}
	}
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

	auto& scoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	double pixToWorld = 0;

	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.xiangSuDangLiang1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		pixToWorld = setConfig.xiangSuDangLiang2;
	}

	for (const auto& item : processIndex)
	{
		ZipperDefectInfo::DetectItem defectItem;

		auto queyaScore = processResult[item].score * 100; // 将分数转换为百分比
		auto queyaArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.queYa && queyaScore >= scoreConfig.queYaScore && queyaArea >= scoreConfig.queYaArea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = queyaScore;
		defectItem.area = queyaArea;
		defectItem.index = static_cast<int>(item);
		info.queYaList.emplace_back(defectItem);
	}
}

void ImageProcessorZipper::getTangshangInfo(ZipperDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}

	auto& scoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	double pixToWorld = 0;

	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.xiangSuDangLiang1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		pixToWorld = setConfig.xiangSuDangLiang2;
	}

	for (const auto& item : processIndex)
	{
		ZipperDefectInfo::DetectItem defectItem;

		auto tangshangScore = processResult[item].score * 100; // 将分数转换为百分比
		auto tangshangArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.tangShang && tangshangScore >= scoreConfig.tangShangScore && tangshangArea >= scoreConfig.tangShangArea)
		{
			defectItem.isDraw = true;
		}
		defectItem.index = static_cast<int>(item);
		info.tangShangList.emplace_back(defectItem);
	}
}

void ImageProcessorZipper::getZangwuInfo(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult,
	const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}

	auto& scoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;

	double pixToWorld = 0;

	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.xiangSuDangLiang1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		pixToWorld = setConfig.xiangSuDangLiang2;
	}

	for (const auto& item : processIndex)
	{
		ZipperDefectInfo::DetectItem defectItem;

		auto zangwuScore = processResult[item].score * 100; // 将分数转换为百分比
		auto zangwuArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.zangWu && zangwuScore >= scoreConfig.zangWuScore && zangwuArea >= scoreConfig.zangWuArea)
		{
			defectItem.isDraw = true;
		}
		defectItem.index = static_cast<int>(item);
		info.zangWuList.emplace_back(defectItem);
	}
}

std::vector<std::vector<size_t>> ImageProcessorZipper::getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info)
{
	std::vector<std::vector<size_t>> result;
	result.resize(20);

	for (int i = 0; i < info.size(); i++)
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
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2));
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
			tangshangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2));
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
			zangwuText.append(QString(" %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2));
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

void ImageProcessorZipper::drawBoundariesLines(QImage& image)
{
	auto& index = imageProcessingModuleIndex;
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;
	rw::rqw::ImagePainter::PainterConfig painterConfig;
	painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Orange);
	if (index == 1)
	{
		rw::rqw::ImagePainter::drawHorizontalLine(image, setConfig.shangXianWei1, painterConfig);
		rw::rqw::ImagePainter::drawHorizontalLine(image, setConfig.xiaXianWei1, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, setConfig.zuoXianWei1, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, setConfig.youXianWei1, painterConfig);
	}
	else if (index == 2)
	{
		rw::rqw::ImagePainter::drawHorizontalLine(image, setConfig.shangXianWei2, painterConfig);
		rw::rqw::ImagePainter::drawHorizontalLine(image, setConfig.xiaXianWei2, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, setConfig.zuoXianWei2, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, setConfig.youXianWei2, painterConfig);
	}
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
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(queyaText);
	}
	// 烫伤
	if (!info.tangShangList.empty()) {
		QString tangshangText("烫伤:");
		for (const auto& item : info.tangShangList) {
			tangshangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(tangshangText);
	}
	// 脏污
	if (!info.zangWuList.empty()) {
		QString zangwuText("脏污:");
		for (const auto& item : info.zangWuList) {
			zangwuText.append(QString(" %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(zangwuText);
	}

	// 显示到左上角
	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList, 0.05);
}

void ImageProcessorZipper::drawDefectRec(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult,
	const std::vector<std::vector<size_t>>& processIndex, const ZipperDefectInfo& info)
{
	if (processResult.size() == 0)
	{
		return;
	}

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);

	// 缺牙
	for (const auto& item : info.queYaList)
	{
		if (!item.isDraw)
		{
			auto& queyaItem = processResult[item.index];
			config.text = QString("缺牙 %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, queyaItem, config);
		}
	}

	// 烫伤
	for (const auto& item : info.tangShangList)
	{
		if (!item.isDraw)
		{
			auto& tangshangItem = processResult[item.index];
			config.text = QString("烫伤 %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, tangshangItem, config);
		}
	}

	// 脏污
	for (const auto& item : info.zangWuList)
	{
		if (!item.isDraw)
		{
			auto& zangwuItem = processResult[item.index];
			config.text = QString("脏污 %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, zangwuItem, config);
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

	// 缺牙
	for (const auto& item : info.queYaList)
	{
		if (item.isDraw)
		{
			auto& queyaItem = processResult[item.index];
			config.text = QString("缺牙 %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, queyaItem, config);
		}
	}

	// 烫伤
	for (const auto& item : info.tangShangList)
	{
		if (item.isDraw)
		{
			auto& tangshangItem = processResult[item.index];
			config.text = QString("烫伤 %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, tangshangItem, config);
		}
	}

	// 脏污
	for (const auto& item : info.zangWuList)
	{
		if (item.isDraw)
		{
			auto& zangwuItem = processResult[item.index];
			config.text = QString("脏污 %1 %2").arg(item.score, 0, 'f', 0).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, zangwuItem, config);
		}
	}
}


void ImageProcessingModuleZipper::onFrameCaptured(cv::Mat frame, size_t index)
{
	if (frame.empty()) {
		return; // 跳过空帧
	}

	auto& globalStruct = GlobalStructDataZipper::getInstance();

	QMutexLocker locker(&_mutex);
	MatInfo mat;
	mat.image = frame;
	mat.index = index;
	mat.location = globalStruct.zmotion.getModbus(0,1);	// 获取拍照的位置
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
		connect(processor, &ImageProcessorZipper::imageNGReady, this, &ImageProcessingModuleZipper::imageNGReady, Qt::QueuedConnection);

		_processors.push_back(processor);
		processor->start();
	}
}

ImageProcessingModuleZipper::ImageProcessingModuleZipper(int numConsumers, QObject* parent)
	: QObject(parent), _numConsumers(numConsumers) {

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
