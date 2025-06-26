#include "ImageProcessorModule.h"

#include <qcolor.h>
#include <QPainter>
#include <random>

#include "GlobalStruct.hpp"
#include"rqw_ImagePainter.h"
#include "Utilty.hpp"
#include"DetachDefectThread.h"
#include "DetachUtiltyThread.h"
#include "imeo_ModelEngineFactory_OnnxRuntime.hpp"
#include"MonitorIO.h"


std::optional<std::chrono::system_clock::time_point> findTimeInterval(
	const std::vector<std::chrono::system_clock::time_point>& timeCollection,
	const std::chrono::system_clock::time_point& timePoint)
{
	if (timeCollection.empty()) {
		return std::nullopt; 
	}

	std::vector<std::chrono::system_clock::time_point> sortedTimes = timeCollection;
	std::sort(sortedTimes.begin(), sortedTimes.end());

	for (size_t i = 0; i < sortedTimes.size() - 1; ++i) {
		if (timePoint >= sortedTimes[i] && timePoint < sortedTimes[i + 1]) {
			return sortedTimes[i];
		}
	}

	return std::nullopt;
}


ImageProcessorSmartCroppingOfBags::ImageProcessorSmartCroppingOfBags(QQueue<MatInfo>& queue, QMutex& mutex, QWaitCondition& condition, int workIndex, QObject* parent)
	: QThread(parent), _queue(queue), _mutex(mutex), _condition(condition), _workIndex(workIndex) {

}

void ImageProcessorSmartCroppingOfBags::run()
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

		auto& globalData = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& globalThreadData = GlobalStructThreadSmartCroppingOfBags::getInstance();

		_qieDaoTime = globalThreadData.currentQieDaoTime;
		_isQieDao = globalThreadData.isQieDao;

		auto currentRunningState = globalData.runningState.load();
		switch (currentRunningState)
		{
		case RunningState::Debug:
			run_debug(frame);
			break;
		case RunningState::openRemove:
			run_OpenRemoveFunc(frame);
			break;
		default:
			break;
		}

	}
}

/*
---
title: 图像处理模块中的debug调用流程
---
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
	start(["给定输入参数 MatInfo& frame"])
	getTimesWithCurrentTime["获取当前时间点和上一次时间点集合，调用_historyTimes->queryWithTime(frame.time,2);"]
	getCurrentWithBeforeTimeCollageTime["获取当前时间点和上一次时间点拼接之后的图像调用_imageCollage->getCollageImage(times);"]
	processCollageImage["AI识别拼接之后的图像调用_modelEngine->processImg(cv::mat)"]
	splitRecognitionBox["将识别的图像分割成两部分，当前时间节点的行高，和上一次的行高"]
	regularizedTwoRecognitionBox["将识别出来的processResult框的集合分别规整到拆分到的两次行高,上也即重新映射到两张图片上"]
	mergeCurrentProcessLastResultWithLastProcessResult["将拆分到的上一次行高的识别框与上一次的行高的识别框的集合直接合并"]
	addCurrentResultToHistoryResult["将拆分后这一次识别的行高的添加到历史的行高识别框中调用_historyResult->inseart()"]
	getCurrentWithBeforeFourTimes["获取当前以及当前之前的供5个时间点"]
	getFiveTimesSouceImage["根据五个时间点获取总共5个原图像"]
	getFiveHistoyProcessResult["根据五个时间点获取总共5个识别信息"]
	drawMaskInfo[分别绘制5个mask图像]
	collageMaskImage[合并绘制之后的图像]

start --> getTimesWithCurrentTime
getTimesWithCurrentTime -->getCurrentWithBeforeTimeCollageTime
getCurrentWithBeforeTimeCollageTime-->processCollageImage
processCollageImage-->splitRecognitionBox
splitRecognitionBox-->regularizedTwoRecognitionBox
regularizedTwoRecognitionBox-->mergeCurrentProcessLastResultWithLastProcessResult
regularizedTwoRecognitionBox-->addCurrentResultToHistoryResult
mergeCurrentProcessLastResultWithLastProcessResult-->getCurrentWithBeforeFourTimes
addCurrentResultToHistoryResult-->getCurrentWithBeforeFourTimes
getCurrentWithBeforeFourTimes-->getFiveTimesSouceImage
getCurrentWithBeforeFourTimes-->getFiveHistoyProcessResult
getFiveTimesSouceImage-->drawMaskInfo
getFiveHistoyProcessResult-->drawMaskInfo
drawMaskInfo-->collageMaskImage
*/
void ImageProcessorSmartCroppingOfBags::run_debug(MatInfo& frame)
{
	// 获得当前图像的时间戳与上一张图像的时间戳的集合
	auto times = getTimesWithCurrentTime_debug(frame.time, 2, true);

	if (times.empty())
	{
		return; // 如果没有时间戳，直接返回
	}

	// 获得当前图像与上一张图像拼接而成的图像
	auto resultImage = getCurrentWithBeforeTimeCollageTime_debug(times);

	auto startTime = std::chrono::high_resolution_clock::now();

	// AI推理获得当前图像与上一张图像拼接而成的图像的检测结果
	auto processResult = processCollageImage_debug(resultImage.mat);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	auto processTime = QString("%1 ms").arg(duration);
	//AI识别完成

	// 获取上一张图像的行高
	auto previousMatHeight = splitRecognitionBox_debug(times);

	// 将识别出来的processResult框的集合分别规整到拆分到的两次行高上,也即重新映射到两张图片上
	regularizedTwoRecognitionBox_debug(previousMatHeight, times[0], frame.time, processResult, processTime);

	auto& globalThreadData = GlobalStructThreadSmartCroppingOfBags::getInstance();
	auto& globalStructData = GlobalStructDataSmartCroppingOfBags::getInstance();

	if (_isQieDao)
	{
		if (frame.time>_qieDaoTime)
		{
			globalThreadData.isQieDao = false;

			//这里第一个时间点可能是上一次的
			auto duringTimes = _historyTimes->query(_lastQieDaoTime,frame.time);

			duringTimes = getValidTime(duringTimes);
			getCutLine(duringTimes, frame);

			// 获取有多少张图片没有拼过
			size_t count = duringTimes.size();

			// 抓取没拼过的图片的时间戳
			auto unprocessedImageTimes = duringTimes;

			if (unprocessedImageTimes.size() != count)
			{
				return; // 如果没有时间戳，直接返回
			}

			// 获取没拼过的图片的原图像
			std::vector<TimeFrameMatInfo> unprocessedimages;

			getUnprocessedSouceImage_debug(unprocessedImageTimes, unprocessedimages);

			// 处理图片及其识别框
			auto fiveQImages = drawUnprocessedMatMaskInfo_debug(unprocessedimages);

			auto collageImage = getCollageImage(fiveQImages);

			//std::cout << "imageReady" << collageImage.size().height() << std::endl;
			emit imageReady(QPixmap::fromImage(collageImage));

			emit appendPixel(collageImage.height());
			//std::cout << "Image emit" << std::endl;
		}
	}
	_lastQieDaoTime = _qieDaoTime;

}

void ImageProcessorSmartCroppingOfBags::drawDebugTextInfoOnQImage(QImage& image, const HistoryDetectInfo& info)
{
	QVector<QString> texts;
	texts.emplace_back(info.processTime);
	std::vector<rw::rqw::ImagePainter::PainterConfig> configs;
	rw::rqw::ImagePainter::PainterConfig config;
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
	config.fontSize = 100;
	configs.emplace_back(config);
	rw::rqw::ImagePainter::drawTextOnImageWithConfig(image, texts, configs);
}

std::vector<Time> ImageProcessorSmartCroppingOfBags::getValidTime(const std::vector<Time>& times)
{
	std::vector<Time> result = times;

	std::vector<Time> itemsToRemove;

	for (size_t i = 0; i < result.size(); ++i)
	{
		if (_timeBool->get(result[i]).has_value())
		{
			if (_timeBool->get(result[i]).value() == true)
			{
				itemsToRemove.push_back(result[i]); // 记录需要删除的元素
			}
			else if (_timeBool->get(result[i]).value() == false)
			{
				_timeBool->set(result[i], true); // 更新状态
			}
		}
	}

	// 删除记录的元素
	for (const auto& item : itemsToRemove)
	{
		result.erase(std::remove(result.begin(), result.end(), item), result.end());
	}

	return result;
}

void ImageProcessorSmartCroppingOfBags::run_monitor(MatInfo& frame)
{

}

std::vector<Time> ImageProcessorSmartCroppingOfBags::getTimesWithCurrentTime_debug(
	const Time& time, int count, bool isBefore, bool ascending)
{
	return _historyTimes->queryWithTime(time, count, isBefore, ascending);
}

ImageCollage::CollageImage ImageProcessorSmartCroppingOfBags::getCurrentWithBeforeTimeCollageTime_debug(
	const std::vector<Time>& times)
{
	auto collageImage = _imageCollage->getCollageImage(times);
	
	return collageImage;
}

std::vector<rw::DetectionRectangleInfo> ImageProcessorSmartCroppingOfBags::processCollageImage_debug(const cv::Mat& mat)
{
	auto result = _modelEngine->processImg(mat);
	return result;
}

int ImageProcessorSmartCroppingOfBags::splitRecognitionBox_debug(const std::vector<Time>& time)
{
	// 获得上个时间戳的cv::Mat图片
	auto previousMat = _imageCollage->getImage(time[0]).value().element;

	// 获取上个时间戳的图像的高度
	auto previousMatHeight = previousMat.rows;

	return previousMatHeight;
}

void ImageProcessorSmartCroppingOfBags::getCutLine(const std::vector<Time>& time, const MatInfo& frame)
{
	if (_isQieDao)
	{
		auto xiangjiDaoDaokouJuli = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig.daokoudaoxiangjiluli1;
		auto daichang= GlobalStructThreadSmartCroppingOfBags::getInstance()._detachUtiltyThreadSmartCroppingOfBags->daichangAverageFromPixel;

		if (daichang<1)
		{
			return;
		}

		auto bagsCount = static_cast<int>(xiangjiDaoDaokouJuli) / static_cast<int>(daichang) < 1 ? 0 : static_cast<int>(xiangjiDaoDaokouJuli) / static_cast<int>(daichang);

		xiangjiDaoDaokouJuli = xiangjiDaoDaokouJuli- daichang*(bagsCount);

		auto cutLocation = xiangjiDaoDaokouJuli/daichang;
		if (cutLocation >= 1)
		{
			return;
		}
		cutLocation = 1 - cutLocation;

		size_t totalImgHeight = 0;
		std::vector<std::pair<Time,size_t>> heights;
		for (const auto & item: time)
		{
			auto img=_imageCollage->getImage(item);
			if (!img.has_value())
			{
				return;
			}
			heights.emplace_back(std::pair<Time,size_t>(item, img.value().element.rows));
			totalImgHeight += img.value().element.rows;
		}

		auto cutLocationXiangSuLocation = totalImgHeight * cutLocation;

		for (int i = 0;i<heights.size();i++)
		{
			if (cutLocationXiangSuLocation > heights[i].second)
			{
				cutLocationXiangSuLocation -= heights[i].second;
			}
			else
			{
				auto cutTimeFrame = heights[i].first;
				auto cutFrameDetectResult=_historyResult->get(cutTimeFrame);
				if (!cutFrameDetectResult.has_value())
				{
					return;
				}
				auto result = cutFrameDetectResult.value();
				result.hasCut = true;
				result.cutLocate = cutLocationXiangSuLocation;
				_historyResult->set(cutTimeFrame, result);
				std::cout << "cutLocate" << result.cutLocate << std::endl;
				return;
			}
		}

	}
}

void ImageProcessorSmartCroppingOfBags::regularizedTwoRecognitionBox_debug(const int& previousMatHeight, const Time& previousTime, const Time& nowTime, std::vector<rw::DetectionRectangleInfo>& allDetectRec, const
                                                                           QString& processTime)
{
	// 将识别出来的processResult框的集合分别规整到拆分到的两次行高上,也即重新映射到两张图片上
	mergeCurrentProcessLastResultWithLastProcessResult_debug(previousMatHeight, previousTime, allDetectRec);

	addCurrentResultToHistoryResult_debug(previousMatHeight, allDetectRec, nowTime, processTime);
}

void ImageProcessorSmartCroppingOfBags::mergeCurrentProcessLastResultWithLastProcessResult_debug(const int& previousMatHeight,const Time& time, std::vector<rw::DetectionRectangleInfo>& allDetectRec)
{
	// 获取上一张图像的检测信息
	auto previousDetectInfo = _historyResult->query(time, 1);

	// 1. 找出属于上一张图片的检测框
	std::vector<rw::DetectionRectangleInfo> belongToPrevious;
	auto it = std::remove_if(allDetectRec.begin(), allDetectRec.end(),
		[previousMatHeight, &belongToPrevious](const rw::DetectionRectangleInfo& rect) {
			bool inPrev = rect.leftTop.second < previousMatHeight &&
				rect.rightTop.second < previousMatHeight &&
				rect.leftBottom.second < previousMatHeight &&
				rect.rightBottom.second < previousMatHeight;
			if (inPrev) {
				belongToPrevious.push_back(rect);
			}
			return inPrev;
		});
	allDetectRec.erase(it, allDetectRec.end());

	// 2. 添加到上一张图片的识别信息
	if (!previousDetectInfo.empty()) {
		previousDetectInfo.front().processResult.insert(
			previousDetectInfo.front().processResult.end(),
			belongToPrevious.begin(),
			belongToPrevious.end()
		);
	}
}

void ImageProcessorSmartCroppingOfBags::addCurrentResultToHistoryResult_debug(const int& previousMatHeight, std::vector<rw::DetectionRectangleInfo>& nowDetectRec, const Time& nowTime, const QString
                                                                              & processTime)
{
	// 剩余检测框的四个顶点y坐标减去上一张图片高度
	for (auto& rect : nowDetectRec) {
		rect.leftTop.second -= previousMatHeight;
		rect.rightTop.second -= previousMatHeight;
		rect.leftBottom.second -= previousMatHeight;
		rect.rightBottom.second -= previousMatHeight;
		rect.center_y -= previousMatHeight;
	}

	HistoryDetectInfo info(nowDetectRec);
	info.processTime = processTime;
	// 将属于当前图片的检测结果添加到当前图像的识别信息中
	_historyResult->insert(nowTime, info);
}

std::vector<Time> ImageProcessorSmartCroppingOfBags::getCurrentWithBeforeFourTimes_debug(
	const Time& time, int count, bool isBefore, bool ascending)
{
	return _historyTimes->queryWithTime(time, count, isBefore, ascending);
}

void ImageProcessorSmartCroppingOfBags::getUnprocessedSouceImage_debug(const std::vector<Time>& fiveTimes, std::vector<TimeFrameMatInfo>& images)
{
	for (const auto & item:fiveTimes)
	{
		auto image = _imageCollage->getImage(item);
		images.emplace_back(TimeFrameMatInfo(item, image));
	}

}

std::vector<TimeFrameQImageInfo> ImageProcessorSmartCroppingOfBags::drawUnprocessedMatMaskInfo_debug(
	const std::vector<TimeFrameMatInfo>& fiveMats)
{
	std::vector<TimeFrameQImageInfo> fiveQImages;
	for (const auto & item : fiveMats)
	{
		if (!item.second.has_value())
		{
			continue;
		}
		auto historyResultItem = _historyResult->get(item.first);
		if (!historyResultItem.has_value())
		{
			continue;;
		}
		auto &historyProcessResult = historyResultItem.value().processResult;
		auto processResultIndex = filterEffectiveIndexes_debug(historyProcessResult);

		SmartCroppingOfBagsDefectInfo defectInfo;
		getEliminationInfo_debug(defectInfo, historyProcessResult, processResultIndex);

		QImage qImage = rw::rqw::cvMatToQImage(item.second.value().element);

		drawDefectRec(qImage, historyProcessResult, processResultIndex, defectInfo);
		drawDefectRec_error(qImage, historyProcessResult, processResultIndex, defectInfo);
		drawDebugTextInfoOnQImage(qImage, historyResultItem.value());

		TimeFrameQImageInfo info(item.first,qImage);
		drawCutLine(info);


		fiveQImages.emplace_back(info);
	}

	return fiveQImages;
}

void ImageProcessorSmartCroppingOfBags::drawCutLine(TimeFrameQImageInfo& info)
{

		auto item=_historyResult->get(info.first);
		if (!item.has_value())
		{
			return;
		}
		auto itemValue = item.value();
		rw::rqw::ImagePainter::PainterConfig config;
		config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
		config.textColor= rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
		config.thickness = 10;
		if (itemValue.hasCut)
		{
			rw::rqw::ImagePainter::drawHorizontalLine(info.second, itemValue.cutLocate, config);
		}

}


QPixmap ImageProcessorSmartCroppingOfBags::collageMaskImage_debug(const QVector<QImage>& fiveQImages)
{
	auto finalImage = _imageCollage->verticalConcat(fiveQImages);

	return QPixmap::fromImage(finalImage);
}

void ImageProcessorSmartCroppingOfBags::getRandomDetecionRec_debug(const ImageCollage::CollageImage& collageImage, std::vector<rw::DetectionRectangleInfo>& detectionRec)
{
	// 随机生成检测框用于测试
	{
		int imgWidth = collageImage.mat.cols;
		int imgHeight = collageImage.mat.rows;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> xDist(0, imgWidth - 60);
		std::uniform_int_distribution<> yDist(0, imgHeight - 60);
		std::uniform_int_distribution<> wDist(30, 100);
		std::uniform_int_distribution<> hDist(30, 100);
		std::uniform_int_distribution<> classDist(0, 5); // 假设有6类
		std::uniform_real_distribution<> scoreDist(0.7, 1.0);

		for (int i = 0; i < 5; ++i) {
			int x = xDist(gen);
			int y = yDist(gen);
			int w = wDist(gen);
			int h = hDist(gen);
			rw::DetectionRectangleInfo rect;
			rect.leftTop = { x, y };
			rect.rightTop = { x + w, y };
			rect.leftBottom = { x, y + h };
			rect.rightBottom = { x + w, y + h };
			rect.center_x = x + w / 2;
			rect.center_y = y + h / 2;
			rect.width = w;
			rect.height = h;
			rect.area = w * h;
			rect.classId = classDist(gen); // 随机类别
			rect.score = scoreDist(gen);   // 随机分数
			detectionRec.push_back(rect);
		}
	}
}

QImage ImageProcessorSmartCroppingOfBags::getCollageImage(const std::vector<TimeFrameQImageInfo>& infos)
{
	QVector<QImage> imgs;
	for (const auto & item:infos)
	{
		imgs.emplace_back(item.second);
	}
	return ImageCollage::verticalConcat(imgs);
}

std::vector<Time> ImageProcessorSmartCroppingOfBags::getTimesWithCurrentTime_Defect(
	const Time& time, int count, bool isBefore, bool ascending)
{
	return _historyTimes->queryWithTime(time, count, isBefore, ascending);
}

ImageCollage::CollageImage ImageProcessorSmartCroppingOfBags::getCurrentWithBeforeTimeCollageTime_Defect(
	const std::vector<Time>& times)
{
	return _imageCollage->getCollageImage(times);
}

std::vector<rw::DetectionRectangleInfo> ImageProcessorSmartCroppingOfBags::processCollageImage_Defect(const cv::Mat& mat)
{
	return _modelEngine->processImg(mat);
}

int ImageProcessorSmartCroppingOfBags::splitRecognitionBox_Defect(
	const std::vector<Time>& time)
{
	// 获得上个时间戳的cv::Mat图片
	auto previousMat = _imageCollage->getImage(time[0]).value().element;

	// 获取上个时间戳的图像的高度
	auto previousMatHeight = previousMat.rows;

	return previousMatHeight;
}

void ImageProcessorSmartCroppingOfBags::regularizedTwoRecognitionBox_Defect(const int& previousMatHeight,
	const Time& previousTime, const Time& nowTime, std::vector<rw::DetectionRectangleInfo>& allDetectRec)
{
	// 将识别出来的processResult框的集合分别规整到拆分到的两次行高上,也即重新映射到两张图片上
	mergeCurrentProcessLastResultWithLastProcessResult_Defect(previousMatHeight, previousTime, allDetectRec);
	addCurrentResultToHistoryResult_Defect(previousMatHeight, allDetectRec, nowTime);
}

void ImageProcessorSmartCroppingOfBags::mergeCurrentProcessLastResultWithLastProcessResult_Defect(
	const int& previousMatHeight, const Time& time, std::vector<rw::DetectionRectangleInfo>& allDetectRec)
{
	// 获取上一张图像的检测信息
	auto previousDetectInfo = _historyResult->query(time, 1);
	// 1. 找出属于上一张图片的检测框
	std::vector<rw::DetectionRectangleInfo> belongToPrevious;
	auto it = std::remove_if(allDetectRec.begin(), allDetectRec.end(),
		[previousMatHeight, &belongToPrevious](const rw::DetectionRectangleInfo& rect) {
			bool inPrev = rect.leftTop.second < previousMatHeight &&
				rect.rightTop.second < previousMatHeight &&
				rect.leftBottom.second < previousMatHeight &&
				rect.rightBottom.second < previousMatHeight;
			if (inPrev) {
				belongToPrevious.push_back(rect);
			}
			return inPrev;
		});
	allDetectRec.erase(it, allDetectRec.end());
	// 2. 添加到上一张图片的识别信息
	if (!previousDetectInfo.empty()) {
		previousDetectInfo.front().processResult.insert(
			previousDetectInfo.front().processResult.end(),
			belongToPrevious.begin(),
			belongToPrevious.end()
		);
	}
}

void ImageProcessorSmartCroppingOfBags::addCurrentResultToHistoryResult_Defect(const int& previousMatHeight,
	std::vector<rw::DetectionRectangleInfo>& nowDetectRec, const Time& nowTime)
{
	// 剩余检测框的四个顶点y坐标减去上一张图片高度
	for (auto& rect : nowDetectRec) {
		rect.leftTop.second -= previousMatHeight;
		rect.rightTop.second -= previousMatHeight;
		rect.leftBottom.second -= previousMatHeight;
		rect.rightBottom.second -= previousMatHeight;
		rect.center_y -= previousMatHeight;
	}
	// 将属于当前图片的检测结果添加到当前图像的识别信息中
	_historyResult->insert(nowTime, HistoryDetectInfo(nowDetectRec));
}

/*
---
title: 图像处理模块中的剔除模块smartCrop调用流程
---
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart TB
    start(["给定输入参数 MatInfo& frame"])
    getTimesWithCurrentTime["获取当前时间点和上一次时间点集合，调用_historyTimes->queryWithTime(frame.time,2);"]
    getCurrentWithBeforeTimeCollageTime["获取当前时间点和上一次时间点拼接之后的图像调用_imageCollage->getCollageImage(times);"]
    processCollageImage["AI识别拼接之后的图像调用_modelEngine->processImg(cv::mat)"]

    getClassIdIndex["获取每一个缺陷的index集合类型std::vector<std::vector<size_t>>"]
    filterValidIndex["获取有效的索引也即将将屏蔽内的识别框过滤掉"]
    getDetectInfo["获取所有的识别信息放到自定义的结构体中"]
    getBottomRec["计算detectInfo自定义的结构体中满足score\窗口中score或其他设置条件的最底部的识别框也即y轴坐标最低,同时将isDraw计算出来"]
    getBottomLocation["将上一步的最底部识别框的y轴坐标转换为location,根据frame.location和脉冲系数去和像素当量算出检测框最底部的location"]
    emitErrorLocation["将错误的location发送到剔废队列中"]

    splitRecognitionBox["将识别的图像分割成两部分，当前时间节点的行高，和上一次的行高"]
    regularizedTwoRecognitionBox["将识别出来的processResult框的集合分别规整到拆分到的两次行高,上也即重新映射到两张图片上"]
    mergeCurrentProcessLastResultWithLastProcessResult["将拆分到的上一次行高的识别框与上一次的行高的识别框的集合直接合并"]
    addCurrentResultToHistoryResult["将拆分后这一次识别的行高的添加到历史的行高识别框中调用_historyResult->inseart()"]
    getCurrentWithBeforeFourTimes["获取当前以及当前之前的供5个时间点"]
    getFiveTimesSouceImage["根据五个时间点获取总共5个原图像"]
    getFiveHistoyProcessResult["根据五个时间点获取总共5个识别信息"]
    drawMaskInfo[分别绘制5个mask图像]
    collageMaskImage[合并绘制之后的图像]
    emitResultImage(["发送绘制好的mask图"])

start --> getTimesWithCurrentTime
getTimesWithCurrentTime -->getCurrentWithBeforeTimeCollageTime
getCurrentWithBeforeTimeCollageTime-->processCollageImage
processCollageImage-->getClassIdIndex
getClassIdIndex-->filterValidIndex
filterValidIndex-->getDetectInfo
getDetectInfo-->getBottomRec
getBottomRec-->getBottomLocation
getBottomLocation-->emitErrorLocation
emitErrorLocation-->splitRecognitionBox
splitRecognitionBox-->regularizedTwoRecognitionBox
regularizedTwoRecognitionBox-->mergeCurrentProcessLastResultWithLastProcessResult
regularizedTwoRecognitionBox-->addCurrentResultToHistoryResult
mergeCurrentProcessLastResultWithLastProcessResult-->getCurrentWithBeforeFourTimes
addCurrentResultToHistoryResult-->getCurrentWithBeforeFourTimes
getCurrentWithBeforeFourTimes-->getFiveTimesSouceImage
getCurrentWithBeforeFourTimes-->getFiveHistoyProcessResult
getFiveTimesSouceImage-->drawMaskInfo
getFiveHistoyProcessResult-->drawMaskInfo
drawMaskInfo-->collageMaskImage
collageMaskImage-->emitResultImage

 */
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc(MatInfo& frame)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& setConfig = globalStruct.setConfig;

	auto times = getTimesWithCurrentTime_Defect(frame.time, 2);

	auto collageImage = getCurrentWithBeforeTimeCollageTime_Defect(times);

	//AI开始识别
	SmartCroppingOfBagsDefectInfo defectInfo;
	auto startTime = std::chrono::high_resolution_clock::now();

	// AI推理获得当前图像与上一张图像拼接而成的图像的检测结果
	auto processResult = processCollageImage_Defect(collageImage.mat);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	//defectInfo.time = QString("处理时间: %1 ms").arg(duration);
	//AI识别完成

	auto previousMatHeight = splitRecognitionBox_Defect(times);

	regularizedTwoRecognitionBox_Defect(previousMatHeight, times[0], frame.time, processResult);

	//过滤出有效索引
	auto processResultIndex = filterEffectiveIndexes_defect(processResult);
	//获取到当前图像的缺陷信息
	getEliminationInfo_defect(defectInfo, processResult, processResultIndex, frame.image.element);


	// 遍历所有缺陷类型，找出isDraw为true的最底部识别框（以下两个顶点的y最大值为准）
	const SmartCroppingOfBagsDefectInfo::DetectItem* bottomItem = nullptr;
	double maxBottomY = std::numeric_limits<double>::min();

	// 所有缺陷类型的列表指针
	std::vector<const std::vector<SmartCroppingOfBagsDefectInfo::DetectItem>*> defectLists = {
		&defectInfo.heibaList, &defectInfo.shudangList, &defectInfo.huapoList, &defectInfo.jietouList,
		&defectInfo.guasiList, &defectInfo.podongList, &defectInfo.zangwuList, &defectInfo.noshudangList,
		&defectInfo.modianList, &defectInfo.loumoList, &defectInfo.xishudangList, &defectInfo.erweimaList,
		&defectInfo.damodianList, &defectInfo.kongdongList, &defectInfo.sebiaoList, &defectInfo.yinshuaquexianList,
		&defectInfo.xiaopodongList, &defectInfo.jiaodaiList
	};

	for (const auto* list : defectLists) {
		for (const auto& item : *list) {
			if (item.isDraw && item.index >= 0 && item.index < processResult.size()) {
				const auto& rect = processResult[item.index];
				double bottomY1 = rect.leftBottom.second;
				double bottomY2 = rect.rightBottom.second;
				double curMaxY = std::max(bottomY1, bottomY2);
				if (curMaxY > maxBottomY) {
					maxBottomY = curMaxY;
					bottomItem = &item;
				}
			}
		}
	}

	// bottomItem即为最底部的识别框（如果存在）
	if (bottomItem) {
		auto nowImageHeight = frame.image.element.rows;

		auto heightRatio = maxBottomY / nowImageHeight; // 计算高度比例

		auto lastImage = _imageCollage->getImage(times[0]).value();

		auto lastImagePulse = lastImage.attribute["location"];

		auto nowImagePulse = frame.location * heightRatio; // 根据高度比例计算位置
		
		auto pulseDifference = nowImagePulse - lastImagePulse;
	}

	// 剔除逻辑获取_isbad以及绘制defect错误信息
	run_OpenRemoveFunc_process_defect_info(defectInfo);
	//如果_isbad为true，将错误信息发送到剔除队列中
	//run_OpenRemoveFunc_emitErrorInfo(frame);
}

void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info(SmartCroppingOfBagsDefectInfo& info)
{
	_isbad = false; // 重置坏品标志
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& isOpenDefect = globalStruct.generalConfig.istifei;

	if (isOpenDefect)
	{
		run_OpenRemoveFunc_process_defect_info_Heiba(info);
		run_OpenRemoveFunc_process_defect_info_Shudang(info);
		run_OpenRemoveFunc_process_defect_info_Huapo(info);
		run_OpenRemoveFunc_process_defect_info_Jietou(info);
		run_OpenRemoveFunc_process_defect_info_Guasi(info);
		run_OpenRemoveFunc_process_defect_info_Podong(info);
		run_OpenRemoveFunc_process_defect_info_Zangwu(info);
		run_OpenRemoveFunc_process_defect_info_Noshudang(info);
		run_OpenRemoveFunc_process_defect_info_Modian(info);
		run_OpenRemoveFunc_process_defect_info_Loumo(info);
		run_OpenRemoveFunc_process_defect_info_Xishudang(info);
		run_OpenRemoveFunc_process_defect_info_Erweima(info);
		run_OpenRemoveFunc_process_defect_info_Damodian(info);
		run_OpenRemoveFunc_process_defect_info_Kongdong(info);
		run_OpenRemoveFunc_process_defect_info_Sebiao(info);
		run_OpenRemoveFunc_process_defect_info_Yinshuaquexian(info);
		run_OpenRemoveFunc_process_defect_info_Xiaopodong(info);
		run_OpenRemoveFunc_process_defect_info_Jiaodai(info);
	}
}

void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Heiba(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.heiba)
	{
		auto& heiba = info.heibaList;
		if (heiba.empty())
		{
			return;
		}
		for (const auto& item : heiba)
		{
			if (item.score >= productScore.heibascore && item.area >= productScore.heibaarea)
			{
				_isbad = true; // 有黑疤就认为是坏品
				break; // 找到一个符合条件的黑疤就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Shudang(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.shudang)
	{
		auto& shudang = info.shudangList;
		if (shudang.empty())
		{
			return;
		}
		for (const auto& item : shudang)
		{
			if (item.score >= productScore.shudangscore && item.area >= productScore.shudangarea)
			{
				_isbad = true; // 有疏档就认为是坏品
				break; // 找到一个符合条件的疏档就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Huapo(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.huapo)
	{
		auto& huapo = info.huapoList;
		if (huapo.empty())
		{
			return;
		}
		for (const auto& item : huapo)
		{
			if (item.score >= productScore.huaposcore && item.area >= productScore.huapoarea)
			{
				_isbad = true; // 有划破就认为是坏品
				break; // 找到一个符合条件的划破就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Jietou(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.jietou)
	{
		auto& jietou = info.jietouList;
		if (jietou.empty())
		{
			return;
		}
		for (const auto& item : jietou)
		{
			if (item.score >= productScore.jietouscore && item.area >= productScore.jietouarea)
			{
				_isbad = true; // 有接头就认为是坏品
				break; // 找到一个符合条件的接头就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Guasi(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.guasi)
	{
		auto& guasi = info.guasiList;
		if (guasi.empty())
		{
			return;
		}
		for (const auto& item : guasi)
		{
			if (item.score >= productScore.guasiscore && item.area >= productScore.guasiarea)
			{
				_isbad = true; // 有挂丝就认为是坏品
				break; // 找到一个符合条件的挂丝就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Podong(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.podong)
	{
		auto& podong = info.podongList;
		if (podong.empty())
		{
			return;
		}
		for (const auto& item : podong)
		{
			if (item.score >= productScore.podongscore && item.area >= productScore.podongarea)
			{
				_isbad = true; // 有破洞就认为是坏品
				break; // 找到一个符合条件的破洞就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Zangwu(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.zangwu)
	{
		auto& zangwu = info.zangwuList;
		if (zangwu.empty())
		{
			return;
		}
		for (const auto& item : zangwu)
		{
			if (item.score >= productScore.zangwuscore && item.area >= productScore.zangwuarea)
			{
				_isbad = true; // 有脏污就认为是坏品
				break; // 找到一个符合条件的脏污就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Noshudang(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.noshudang)
	{
		auto& noshudang = info.noshudangList;
		if (noshudang.empty())
		{
			return;
		}
		for (const auto& item : noshudang)
		{
			if (item.score >= productScore.noshudangscore && item.area >= productScore.noshudangarea)
			{
				_isbad = true; // 有无疏档就认为是坏品
				break; // 找到一个符合条件的无疏档就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Modian(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.modian)
	{
		auto& modian = info.modianList;
		if (modian.empty())
		{
			return;
		}
		for (const auto& item : modian)
		{
			if (item.score >= productScore.modianscore && item.area >= productScore.modianarea)
			{
				_isbad = true; // 有磨点就认为是坏品
				break; // 找到一个符合条件的磨点就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Loumo(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.loumo)
	{
		auto& loumo = info.loumoList;
		if (loumo.empty())
		{
			return;
		}
		for (const auto& item : loumo)
		{
			if (item.score >= productScore.loumoscore && item.area >= productScore.loumoarea)
			{
				_isbad = true; // 有漏膜就认为是坏品
				break; // 找到一个符合条件的漏膜就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Xishudang(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.xishudang)
	{
		auto& xishudang = info.xishudangList;
		if (xishudang.empty())
		{
			return;
		}
		for (const auto& item : xishudang)
		{
			if (item.score >= productScore.xishudangscore && item.area >= productScore.xishudangarea)
			{
				_isbad = true; // 有细疏档就认为是坏品
				break; // 找到一个符合条件的细疏档就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Erweima(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.erweima)
	{
		auto& erweima = info.erweimaList;
		if (erweima.empty())
		{
			return;
		}
		for (const auto& item : erweima)
		{
			if (item.score >= productScore.erweimascore && item.area >= productScore.erweimaarea)
			{
				_isbad = true; // 有二维码就认为是坏品
				break; // 找到一个符合条件的二维码就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Damodian(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.damodian)
	{
		auto& damodian = info.damodianList;
		if (damodian.empty())
		{
			return;
		}
		for (const auto& item : damodian)
		{
			if (item.score >= productScore.damodianscore && item.area >= productScore.damodianarea)
			{
				_isbad = true; // 有大墨点就认为是坏品
				break; // 找到一个符合条件的大墨点就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Kongdong(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.kongdong)
	{
		auto& kongdong = info.kongdongList;
		if (kongdong.empty())
		{
			return;
		}
		for (const auto& item : kongdong)
		{
			if (item.score >= productScore.kongdongscore && item.area >= productScore.kongdongarea)
			{
				_isbad = true; // 有孔洞就认为是坏品
				break; // 找到一个符合条件的孔洞就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Sebiao(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.sebiao)
	{
		auto& sebiao = info.sebiaoList;
		if (sebiao.empty())
		{
			return;
		}
		for (const auto& item : sebiao)
		{
			if (item.score >= productScore.sebiaoscore && item.area >= productScore.sebiaoarea)
			{
				_isbad = true; // 有色标就认为是坏品
				break; // 找到一个符合条件的色标就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Yinshuaquexian(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.yinshuaquexian)
	{
		auto& yinshuaquexian = info.yinshuaquexianList;
		if (yinshuaquexian.empty())
		{
			return;
		}
		for (const auto& item : yinshuaquexian)
		{
			if (item.score >= productScore.yinshuaquexianscore && item.area >= productScore.yinshuaquexianarea)
			{
				_isbad = true; // 有印刷缺陷就认为是坏品
				break; // 找到一个符合条件的印刷缺陷就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Xiaopodong(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.xiaopodong)
	{
		auto& xiaopodong = info.xiaopodongList;
		if (xiaopodong.empty())
		{
			return;
		}
		for (const auto& item : xiaopodong)
		{
			if (item.score >= productScore.xiaopodongscore && item.area >= productScore.xiaopodongarea)
			{
				_isbad = true; // 有小破洞就认为是坏品
				break; // 找到一个符合条件的小破洞就可以了
			}
		}
	}
}
void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_process_defect_info_Jiaodai(
	SmartCroppingOfBagsDefectInfo& info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& productScore = globalStruct.scoreConfig;
	if (productScore.jiaodai)
	{
		auto& jiaodai = info.jiaodaiList;
		if (jiaodai.empty())
		{
			return;
		}
		for (const auto& item : jiaodai)
		{
			if (item.score >= productScore.jiaodaiscore && item.area >= productScore.jiaodaiarea)
			{
				_isbad = true; // 有胶带就认为是坏品
				break; // 找到一个符合条件的胶带就可以了
			}
		}
	}
}

void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	if (_isbad)
	{
		++globalStruct.statisticalInfo.wasteCount;
	}

	if (imageProcessingModuleIndex == 1 || imageProcessingModuleIndex == 2)
	{
		++globalStruct.statisticalInfo.produceCount;
	}


	/*if (_isbad)
	{
		switch (imageProcessingModuleIndex)
		{
		case 1:
			globalStruct.priorityQueue1->insert(frame.time, frame.time);
			break;
		case 2:
			globalStruct.priorityQueue2->insert(frame.time, frame.time);
			break;
		default:
			break;
		}
	}*/
}

void ImageProcessorSmartCroppingOfBags::save_image(rw::rqw::ImageInfo& imageInfo, const QImage& image)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& generalConfig = globalStruct.generalConfig;

	if (!globalStruct.isTakePictures)
	{
		return;
	}

	if (imageProcessingModuleIndex == 1 && generalConfig.iscuntu)
	{
		if (globalStruct.isTakePictures) {
			save_image_work(imageInfo, image);
		}
	}
	else if (imageProcessingModuleIndex == 2 && generalConfig.iscuntu)
	{
		if (globalStruct.isTakePictures) {
			save_image_work(imageInfo, image);
		}
	}
}

void ImageProcessorSmartCroppingOfBags::save_image_work(rw::rqw::ImageInfo& imageInfo, const QImage& image)
{
	auto& globalData = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& setConfig = globalData.setConfig;
	if (_isbad) {
		if (true)
		{
			imageInfo.classify = "NG";
			globalData.imageSaveEngine->pushImage(imageInfo);
		}
		if (true)
		{
			rw::rqw::ImageInfo mask(image);
			mask.classify = "Mask";
			globalData.imageSaveEngine->pushImage(mask);
		}
	}
	else {
		if (true)
		{
			imageInfo.classify = "OK";
			globalData.imageSaveEngine->pushImage(imageInfo);
		}
	}
}

//void ImageProcessorSmartCroppingOfBags::monitorIO()
//{
//	auto& motion = GlobalStructDataSmartCroppingOfBags::getInstance().motion;
//	while (true)
//	{
//
//		//运动控制卡获得io
//		bool nowstate = motion->GetIOIn(0);
//		//上升延
//		//说明下切刀了
//		//出图信号
//		//或得这个时间点的脉冲信号,location
//		double nowlocation = location;
//
//		if (nowstate == true && state == false)
//		{
//
//
//			//拼接前面所有图片为一张
//
//
//
//			//求5个袋子的平均袋长
//			//1通过像素求
//			//2通过脉冲去求
//
//
//
//
//
//			//在这张图片上画缺陷
//
//			//画切刀线
//
//
//			//进行一次判断,判断这个袋子上面是否有缺陷
//
//
//
//
//
//
//
//
//		}
//
//		state = nowstate;
//
//	}
//}

void ImageProcessorSmartCroppingOfBags::getEliminationInfo_debug(SmartCroppingOfBagsDefectInfo& info,
                                                                 const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index)
{
	getHeibaInfo(info, processResult, index[ClassId::Heiba]);
	getShudangInfo(info, processResult, index[ClassId::Shudang]);
	getHuapoInfo(info, processResult, index[ClassId::Huapo]);
	getJietouInfo(info, processResult, index[ClassId::Jietou]);
	getGuasiInfo(info, processResult, index[ClassId::Guasi]);
	getPodongInfo(info, processResult, index[ClassId::Podong]);
	getZangwuInfo(info, processResult, index[ClassId::Zangwu]);
	getNoshudangInfo(info, processResult, index[ClassId::Noshudang]);
	getModianInfo(info, processResult, index[ClassId::Modian]);
	getLoumoInfo(info, processResult, index[ClassId::Loumo]);
	getXishudangInfo(info, processResult, index[ClassId::Xishudang]);
	getErweimaInfo(info, processResult, index[ClassId::Erweima]);
	getDamodianInfo(info, processResult, index[ClassId::Damodian]);
	getKongdongInfo(info, processResult, index[ClassId::Kongdong]);
	getSebiaoInfo(info, processResult, index[ClassId::Sebiao]);
	getYinshuaquexianInfo(info, processResult, index[ClassId::Yinshuaquexian]);
	getXiaopodongInfo(info, processResult, index[ClassId::Xiaopodong]);
	getJiaodaiInfo(info, processResult, index[ClassId::Jiaodai]);
}

void ImageProcessorSmartCroppingOfBags::getEliminationInfo_defect(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index,
	const cv::Mat& mat)
{
	getHeibaInfo(info, processResult, index[ClassId::Heiba]);
	getShudangInfo(info, processResult, index[ClassId::Shudang]);
	getHuapoInfo(info, processResult, index[ClassId::Huapo]);
	getJietouInfo(info, processResult, index[ClassId::Jietou]);
	getGuasiInfo(info, processResult, index[ClassId::Guasi]);
	getPodongInfo(info, processResult, index[ClassId::Podong]);
	getZangwuInfo(info, processResult, index[ClassId::Zangwu]);
	getNoshudangInfo(info, processResult, index[ClassId::Noshudang]);
	getModianInfo(info, processResult, index[ClassId::Modian]);
	getLoumoInfo(info, processResult, index[ClassId::Loumo]);
	getXishudangInfo(info, processResult, index[ClassId::Xishudang]);
	getErweimaInfo(info, processResult, index[ClassId::Erweima]);
	getDamodianInfo(info, processResult, index[ClassId::Damodian]);
	getKongdongInfo(info, processResult, index[ClassId::Kongdong]);
	getSebiaoInfo(info, processResult, index[ClassId::Sebiao]);
	getYinshuaquexianInfo(info, processResult, index[ClassId::Yinshuaquexian]);
	getXiaopodongInfo(info, processResult, index[ClassId::Xiaopodong]);
	getJiaodaiInfo(info, processResult, index[ClassId::Jiaodai]);
}

void ImageProcessorSmartCroppingOfBags::getHeibaInfo(SmartCroppingOfBagsDefectInfo& info,
                                                     const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}

	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;

	double pixToWorld = 0;

	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}

	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;

		auto heibaScore = processResult[item].score * 100; // 将分数转换为百分比
		auto heibaArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.heiba && heibaScore >= scoreConfig.heibascore && heibaArea >= scoreConfig.heibaarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = heibaScore;
		defectItem.area = heibaArea;
		defectItem.index = static_cast<int>(item);
		info.heibaList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getShudangInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto shudangScore = processResult[item].score * 100; // 将分数转换为百分比
		auto shudangArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.shudang && shudangScore >= scoreConfig.shudangscore && shudangArea >= scoreConfig.shudangarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = shudangScore;
		defectItem.area = shudangArea;
		defectItem.index = static_cast<int>(item);
		info.shudangList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getHuapoInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto huapoScore = processResult[item].score * 100; // 将分数转换为百分比
		auto huapoArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.huapo && huapoScore >= scoreConfig.huaposcore && huapoArea >= scoreConfig.huapoarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = huapoScore;
		defectItem.area = huapoArea;
		defectItem.index = static_cast<int>(item);
		info.huapoList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getJietouInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto jietouScore = processResult[item].score * 100; // 将分数转换为百分比
		auto jietouArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.jietou && jietouScore >= scoreConfig.jietouscore && jietouArea >= scoreConfig.jietouarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = jietouScore;
		defectItem.area = jietouArea;
		defectItem.index = static_cast<int>(item);
		info.jietouList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getGuasiInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto guasiScore = processResult[item].score * 100; // 将分数转换为百分比
		auto guasiArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.guasi && guasiScore >= scoreConfig.guasiscore && guasiArea >= scoreConfig.guasiarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = guasiScore;
		defectItem.area = guasiArea;
		defectItem.index = static_cast<int>(item);
		info.guasiList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getPodongInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto podongScore = processResult[item].score * 100; // 将分数转换为百分比
		auto podongArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.podong && podongScore >= scoreConfig.podongscore && podongArea >= scoreConfig.podongarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = podongScore;
		defectItem.area = podongArea;
		defectItem.index = static_cast<int>(item);
		info.podongList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getZangwuInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto zangwuScore = processResult[item].score * 100; // 将分数转换为百分比
		auto zangwuArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.zangwu && zangwuScore >= scoreConfig.zangwuscore && zangwuArea >= scoreConfig.zangwuarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = zangwuScore;
		defectItem.area = zangwuArea;
		defectItem.index = static_cast<int>(item);
		info.zangwuList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getNoshudangInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto noshudangScore = processResult[item].score * 100; // 将分数转换为百分比
		auto noshudangArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.noshudang && noshudangScore >= scoreConfig.noshudangscore && noshudangArea >= scoreConfig.noshudangarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = noshudangScore;
		defectItem.area = noshudangArea;
		defectItem.index = static_cast<int>(item);
		info.noshudangList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getModianInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto modianScore = processResult[item].score * 100; // 将分数转换为百分比
		auto modianArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.modian && modianScore >= scoreConfig.modianscore && modianArea >= scoreConfig.modianarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = modianScore;
		defectItem.area = modianArea;
		defectItem.index = static_cast<int>(item);
		info.modianList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getLoumoInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto loumoScore = processResult[item].score * 100; // 将分数转换为百分比
		auto loumoArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.loumo && loumoScore >= scoreConfig.loumoscore && loumoArea >= scoreConfig.loumoarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = loumoScore;
		defectItem.area = loumoArea;
		defectItem.index = static_cast<int>(item);
		info.loumoList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getXishudangInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto xishudangScore = processResult[item].score * 100; // 将分数转换为百分比
		auto xishudangArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.xishudang && xishudangScore >= scoreConfig.xishudangscore && xishudangArea >= scoreConfig.xishudangarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = xishudangScore;
		defectItem.area = xishudangArea;
		defectItem.index = static_cast<int>(item);
		info.xishudangList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getErweimaInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto erweimaScore = processResult[item].score * 100; // 将分数转换为百分比
		auto erweimaArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.erweima && erweimaScore >= scoreConfig.erweimascore && erweimaArea >= scoreConfig.erweimaarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = erweimaScore;
		defectItem.area = erweimaArea;
		defectItem.index = static_cast<int>(item);
		info.erweimaList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getDamodianInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto damodianScore = processResult[item].score * 100; // 将分数转换为百分比
		auto damodianArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.damodian && damodianScore >= scoreConfig.damodianscore && damodianArea >= scoreConfig.damodianarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = damodianScore;
		defectItem.area = damodianArea;
		defectItem.index = static_cast<int>(item);
		info.damodianList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getKongdongInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto kongdongScore = processResult[item].score * 100; // 将分数转换为百分比
		auto kongdongArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.kongdong && kongdongScore >= scoreConfig.kongdongscore && kongdongArea >= scoreConfig.kongdongarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = kongdongScore;
		defectItem.area = kongdongArea;
		defectItem.index = static_cast<int>(item);
		info.kongdongList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getSebiaoInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto sebiaoScore = processResult[item].score * 100; // 将分数转换为百分比
		auto sebiaoArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.sebiao && sebiaoScore >= scoreConfig.sebiaoscore && sebiaoArea >= scoreConfig.sebiaoarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = sebiaoScore;
		defectItem.area = sebiaoArea;
		defectItem.index = static_cast<int>(item);
		info.sebiaoList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getYinshuaquexianInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto yinshuaquexianScore = processResult[item].score * 100; // 将分数转换为百分比
		auto yinshuaquexianArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.yinshuaquexian && yinshuaquexianScore >= scoreConfig.yinshuaquexianscore &&
			yinshuaquexianArea >= scoreConfig.yinshuaquexianarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = yinshuaquexianScore;
		defectItem.area = yinshuaquexianArea;
		defectItem.index = static_cast<int>(item);
		info.yinshuaquexianList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getXiaopodongInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto xiaopodongScore = processResult[item].score * 100; // 将分数转换为百分比
		auto xiaopodongArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.xiaopodong && xiaopodongScore >= scoreConfig.xiaopodongscore &&
			xiaopodongArea >= scoreConfig.xiaopodongarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = xiaopodongScore;
		defectItem.area = xiaopodongArea;
		defectItem.index = static_cast<int>(item);
		info.xiaopodongList.emplace_back(defectItem);
	}
}

void ImageProcessorSmartCroppingOfBags::getJiaodaiInfo(SmartCroppingOfBagsDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	double pixToWorld = 0;
	if (imageProcessingModuleIndex == 1)
	{
		pixToWorld = setConfig.daichangxishu1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		//pixToWorld = setConfig.daichangxishu2;
	}
	for (const auto& item : processIndex)
	{
		SmartCroppingOfBagsDefectInfo::DetectItem defectItem;
		auto jiaodaiScore = processResult[item].score * 100; // 将分数转换为百分比
		auto jiaodaiArea = static_cast<double>(processResult[item].area * pixToWorld * pixToWorld); // 获取面积
		if (scoreConfig.jiaodai && jiaodaiScore >= scoreConfig.jiaodaiscore && jiaodaiArea >= scoreConfig.jiaodaiarea)
		{
			defectItem.isDraw = true;
		}
		defectItem.score = jiaodaiScore;
		defectItem.area = jiaodaiArea;
		defectItem.index = static_cast<int>(item);
		info.jiaodaiList.emplace_back(defectItem);
	}
}

std::vector<std::vector<size_t>> ImageProcessorSmartCroppingOfBags::getClassIndex(
	const std::vector<rw::DetectionRectangleInfo>& info)
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

void ImageProcessorSmartCroppingOfBags::buildSegModelEngine(const QString& enginePath)
{
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.1f;
	config.nms_threshold = 0.1f;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::LetterBox;
	config.letterBoxColor = cv::Scalar(114, 114, 114);
	config.modelPath = enginePath.toStdString();
	_modelEngine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::Yolov11_Det, rw::ModelEngineDeployType::TensorRT);
}

std::vector<std::vector<size_t>> ImageProcessorSmartCroppingOfBags::filterEffectiveIndexes_debug(
	std::vector<rw::DetectionRectangleInfo> info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	auto processIndex = getClassIndex(info);
	processIndex = getIndexInBoundary(info, processIndex);
	return processIndex;
}

std::vector<std::vector<size_t>> ImageProcessorSmartCroppingOfBags::filterEffectiveIndexes_defect(
	std::vector<rw::DetectionRectangleInfo> info)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	auto processIndex = getClassIndex(info);
	//processIndex = getIndexInBoundary(info, processIndex);
	return processIndex;
}

std::vector<std::vector<size_t>> ImageProcessorSmartCroppingOfBags::getIndexInBoundary(
	const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index)
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

bool ImageProcessorSmartCroppingOfBags::isInBoundary(const rw::DetectionRectangleInfo& info)
{
	return true;
}

void ImageProcessorSmartCroppingOfBags::drawSmartCroppingOfBagsDefectInfoText_defect(QImage& image,
	const SmartCroppingOfBagsDefectInfo& info)
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
	//textList.push_back(info.time);

	auto& generalSet = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	auto& isDefect = generalSet.istifei;

	// 如果开启了剔废功能
	if (isDefect)
	{
		// 添加剔废信息(如果信息内容太多记得修改)
		appendHeibaDectInfo(textList, info);
		appendShudangDectInfo(textList, info);
		appendHuapoDectInfo(textList, info);
		appendJietouDectInfo(textList, info);
		appendGuasiDectInfo(textList, info);
		appendPodongDectInfo(textList, info);
		appendZangwuDectInfo(textList, info);
		appendNoshudangDectInfo(textList, info);
		appendModianDectInfo(textList, info);
		appendLoumoDectInfo(textList, info);
		appendXishudangDectInfo(textList, info);
		appendErweimaDectInfo(textList, info);
		appendDamodianDectInfo(textList, info);
		appendKongdongDectInfo(textList, info);
		appendSebiaoDectInfo(textList, info);
		appendYinshuaquexianDectInfo(textList, info);
		appendXiaopodongDectInfo(textList, info);
		appendJiaodaiDectInfo(textList, info);
	}

	// 将信息显示到左上角
	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList);
}

void ImageProcessorSmartCroppingOfBags::appendHeibaDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.heiba && !info.heibaList.empty())
	{
		QString queyaText("黑疤:");
		for (const auto& item : info.heibaList)
		{
			queyaText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		queyaText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.heibascore)).arg(static_cast<int>(productScore.heibaarea)));
		textList.push_back(queyaText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendShudangDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.shudang && !info.shudangList.empty())
	{
		QString shudangText("疏档:");
		for (const auto& item : info.shudangList)
		{
			shudangText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		shudangText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.shudangscore)).arg(static_cast<int>(productScore.shudangarea)));
		textList.push_back(shudangText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendHuapoDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.huapo && !info.huapoList.empty())
	{
		QString huapoText("划破:");
		for (const auto& item : info.huapoList)
		{
			huapoText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		huapoText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.huaposcore)).arg(static_cast<int>(productScore.huapoarea)));
		textList.push_back(huapoText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendJietouDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.jietou && !info.jietouList.empty())
	{
		QString jietouText("接头:");
		for (const auto& item : info.jietouList)
		{
			jietouText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		jietouText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.jietouscore)).arg(static_cast<int>(productScore.jietouarea)));
		textList.push_back(jietouText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendGuasiDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.guasi && !info.guasiList.empty())
	{
		QString guasiText("挂丝:");
		for (const auto& item : info.guasiList)
		{
			guasiText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		guasiText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.guasiscore)).arg(static_cast<int>(productScore.guasiarea)));
		textList.push_back(guasiText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendPodongDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.podong && !info.podongList.empty())
	{
		QString podongText("破洞:");
		for (const auto& item : info.podongList)
		{
			podongText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		podongText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.podongscore)).arg(static_cast<int>(productScore.podongarea)));
		textList.push_back(podongText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendZangwuDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.zangwu && !info.zangwuList.empty())
	{
		QString zangwuText("脏污:");
		for (const auto& item : info.zangwuList)
		{
			zangwuText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		zangwuText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.zangwuscore)).arg(static_cast<int>(productScore.zangwuarea)));
		textList.push_back(zangwuText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendNoshudangDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.noshudang && !info.noshudangList.empty())
	{
		QString noshudangText("无疏档:");
		for (const auto& item : info.noshudangList)
		{
			noshudangText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		noshudangText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.noshudangscore)).arg(static_cast<int>(productScore.noshudangarea)));
		textList.push_back(noshudangText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendModianDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.modian && !info.modianList.empty())
	{
		QString modianText("墨点:");
		for (const auto& item : info.modianList)
		{
			modianText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		modianText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.modianscore)).arg(static_cast<int>(productScore.modianarea)));
		textList.push_back(modianText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendLoumoDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.loumo && !info.loumoList.empty())
	{
		QString loumoText("漏膜:");
		for (const auto& item : info.loumoList)
		{
			loumoText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		loumoText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.loumoscore)).arg(static_cast<int>(productScore.loumoarea)));
		textList.push_back(loumoText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendXishudangDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.xishudang && !info.xishudangList.empty())
	{
		QString xishudangText("稀疏档:");
		for (const auto& item : info.xishudangList)
		{
			xishudangText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		xishudangText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.xishudangscore)).arg(static_cast<int>(productScore.xishudangarea)));
		textList.push_back(xishudangText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendErweimaDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.erweima && !info.erweimaList.empty())
	{
		QString erweimaText("二维码:");
		for (const auto& item : info.erweimaList)
		{
			erweimaText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		erweimaText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.erweimascore)).arg(static_cast<int>(productScore.erweimaarea)));
		textList.push_back(erweimaText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendDamodianDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.damodian && !info.damodianList.empty())
	{
		QString damodianText("大墨点:");
		for (const auto& item : info.damodianList)
		{
			damodianText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		damodianText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.damodianscore)).arg(static_cast<int>(productScore.damodianarea)));
		textList.push_back(damodianText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendKongdongDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.kongdong && !info.kongdongList.empty())
	{
		QString kongdongText("孔洞:");
		for (const auto& item : info.kongdongList)
		{
			kongdongText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		kongdongText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.kongdongscore)).arg(static_cast<int>(productScore.kongdongarea)));
		textList.push_back(kongdongText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendSebiaoDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.sebiao && !info.sebiaoList.empty())
	{
		QString sebiaoText("色标:");
		for (const auto& item : info.sebiaoList)
		{
			sebiaoText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		sebiaoText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.sebiaoscore)).arg(static_cast<int>(productScore.sebiaoarea)));
		textList.push_back(sebiaoText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendYinshuaquexianDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.yinshuaquexian && !info.yinshuaquexianList.empty())
	{
		QString yinshuaquexianText("印刷缺陷:");
		for (const auto& item : info.yinshuaquexianList)
		{
			yinshuaquexianText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		yinshuaquexianText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.yinshuaquexianscore)).arg(static_cast<int>(productScore.yinshuaquexianarea)));
		textList.push_back(yinshuaquexianText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendXiaopodongDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.xiaopodong && !info.xiaopodongList.empty())
	{
		QString xiaopodongText("小破洞:");
		for (const auto& item : info.xiaopodongList)
		{
			xiaopodongText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		xiaopodongText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.xiaopodongscore)).arg(static_cast<int>(productScore.xiaopodongarea)));
		textList.push_back(xiaopodongText);
	}
}

void ImageProcessorSmartCroppingOfBags::appendJiaodaiDectInfo(QVector<QString>& textList,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& productScore = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	if (_isbad && productScore.jiaodai && !info.jiaodaiList.empty())
	{
		QString jiaodaiText("胶带:");
		for (const auto& item : info.jiaodaiList)
		{
			jiaodaiText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		jiaodaiText.append(QString(" 目标分数: %1,目标面积: %2").arg(static_cast<int>(productScore.jiaodaiscore)).arg(static_cast<int>(productScore.jiaodaiarea)));
		textList.push_back(jiaodaiText);
	}
}

void ImageProcessorSmartCroppingOfBags::drawVerticalLine_locate(QImage& image, size_t locate)
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

void ImageProcessorSmartCroppingOfBags::drawBoundariesLines(QImage& image)
{
	auto& index = imageProcessingModuleIndex;
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	rw::rqw::ImagePainter::PainterConfig painterConfig;
	painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Orange);

	if (index == 1)
	{
		//rw::rqw::ImagePainter::drawHorizontalLine(image, setConfig.shangXianWei1, painterConfig);
	}
	if (index == 2)
	{
		//rw::rqw::ImagePainter::drawHorizontalLine(image, setConfig.shangXianWei1, painterConfig);
	}

}

void ImageProcessorSmartCroppingOfBags::drawSmartCroppingOfBagsDefectInfoText_Debug(QImage& image,
	const SmartCroppingOfBagsDefectInfo& info)
{
	QVector<QString> textList;
	std::vector<rw::rqw::ImagePainter::PainterConfig> configList;
	rw::rqw::ImagePainter::PainterConfig config;

	configList.push_back(config);
	//运行时间
	//textList.push_back(info.time);

	// 黑疤
	if (!info.heibaList.empty()) {
		QString queyaText("黑疤:");
		for (const auto& item : info.heibaList) {
			queyaText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(queyaText);
	}
	// 疏档
	if (!info.shudangList.empty()) {
		QString shudangText("疏档:");
		for (const auto& item : info.shudangList) {
			shudangText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(shudangText);
	}
	// 划破
	if (!info.huapoList.empty()) {
		QString huapoText("划破:");
		for (const auto& item : info.huapoList) {
			huapoText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(huapoText);
	}
	// 接头
	if (!info.jietouList.empty()) {
		QString jietouText("接头:");
		for (const auto& item : info.jietouList) {
			jietouText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(jietouText);
	}
	// 挂丝
	if (!info.guasiList.empty()) {
		QString guasiText("挂丝:");
		for (const auto& item : info.guasiList) {
			guasiText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(guasiText);
	}
	// 破洞
	if (!info.podongList.empty()) {
		QString guasiText("破洞:");
		for (const auto& item : info.podongList) {
			guasiText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(guasiText);
	}
	// 脏污
	if (!info.zangwuList.empty()) {
		QString zangwuText("脏污:");
		for (const auto& item : info.zangwuList) {
			zangwuText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(zangwuText);
	}
	// 无疏档
	if (!info.noshudangList.empty()) {
		QString noshudangText("无疏档:");
		for (const auto& item : info.noshudangList) {
			noshudangText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(noshudangText);
	}
	// 墨点
	if (!info.modianList.empty()) {
		QString modianText("墨点:");
		for (const auto& item : info.modianList) {
			modianText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(modianText);
	}
	// 漏膜
	if (!info.loumoList.empty()) {
		QString loumoText("漏膜:");
		for (const auto& item : info.loumoList) {
			loumoText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(loumoText);
	}
	// 稀疏档
	if (!info.xishudangList.empty()) {
		QString xishudangText("稀疏档:");
		for (const auto& item : info.xishudangList) {
			xishudangText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(xishudangText);
	}
	// 二维码
	if (!info.erweimaList.empty()) {
		QString erweimaText("二维码:");
		for (const auto& item : info.erweimaList) {
			erweimaText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(erweimaText);
	}
	// 大墨点
	if (!info.damodianList.empty()) {
		QString damodianText("大墨点:");
		for (const auto& item : info.damodianList) {
			damodianText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(damodianText);
	}
	// 孔洞
	if (!info.kongdongList.empty()) {
		QString kongdongText("孔洞:");
		for (const auto& item : info.kongdongList) {
			kongdongText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(kongdongText);
	}
	// 色标
	if (!info.sebiaoList.empty()) {
		QString sebiaoText("色标:");
		for (const auto& item : info.sebiaoList) {
			sebiaoText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(sebiaoText);
	}
	// 印刷缺陷
	if (!info.yinshuaquexianList.empty()) {
		QString yinshuaquexianText("印刷缺陷:");
		for (const auto& item : info.yinshuaquexianList) {
			yinshuaquexianText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(yinshuaquexianText);
	}
	// 小破洞
	if (!info.xiaopodongList.empty()) {
		QString xiaopodongText("小破洞:");
		for (const auto& item : info.xiaopodongList) {
			xiaopodongText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(xiaopodongText);
	}
	// 胶带
	if (!info.jiaodaiList.empty()) {
		QString jiaodaiText("胶带:");
		for (const auto& item : info.jiaodaiList) {
			jiaodaiText.append(QString(" %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area)));
		}
		textList.push_back(jiaodaiText);
	}

	// 显示到左上角
	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList, 0.05);
}

void ImageProcessorSmartCroppingOfBags::drawDefectRec(QImage& image,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex,
	const SmartCroppingOfBagsDefectInfo& info)
{
	if (processResult.size() == 0)
	{
		return;
	}

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.fontSize = 50;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);

	auto& daichangxishu = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig.daichangxishu1;
	auto daichangxishuSqrt = daichangxishu * daichangxishu;
	// 黑疤
	for (const auto& item : info.heibaList)
	{
		if (!item.isDraw)
		{
			auto& heibaItem = processResult[item.index];
			config.text = QString("黑疤 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area* daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, heibaItem, config);
		}
	}
	// 疏档
	for (const auto& item : info.shudangList)
	{
		if (!item.isDraw)
		{
			auto& shudangItem = processResult[item.index];
			config.text = QString("疏档 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, shudangItem, config);
		}
	}
	// 划破
	for (const auto& item : info.huapoList)
	{
		if (!item.isDraw)
		{
			auto& huapoItem = processResult[item.index];
			config.text = QString("划破 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, huapoItem, config);
		}
	}
	// 接头
	for (const auto& item : info.jietouList)
	{
		if (!item.isDraw)
		{
			auto& jietouItem = processResult[item.index];
			config.text = QString("接头 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jietouItem, config);
		}
	}
	// 挂丝
	for (const auto& item : info.guasiList)
	{
		if (!item.isDraw)
		{
			auto& guasiItem = processResult[item.index];
			config.text = QString("挂丝 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, guasiItem, config);
		}
	}
	// 破洞
	for (const auto& item : info.podongList)
	{
		if (!item.isDraw)
		{
			auto& podongItem = processResult[item.index];
			config.text = QString("破洞 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, podongItem, config);
		}
	}
	// 脏污
	for (const auto& item : info.zangwuList)
	{
		if (!item.isDraw)
		{
			auto& zangwuItem = processResult[item.index];
			config.text = QString("脏污 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, zangwuItem, config);
		}
	}
	// 无疏档
	for (const auto& item : info.noshudangList)
	{
		if (!item.isDraw)
		{
			auto& noshudangItem = processResult[item.index];
			config.text = QString("无疏档 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, noshudangItem, config);
		}
	}
	// 墨点
	for (const auto& item : info.modianList)
	{
		if (!item.isDraw)
		{
			auto& modianItem = processResult[item.index];
			config.text = QString("墨点 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, modianItem, config);
		}
	}
	// 漏膜
	for (const auto& item : info.loumoList)
	{
		if (!item.isDraw)
		{
			auto& loumoItem = processResult[item.index];
			config.text = QString("漏膜 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, loumoItem, config);
		}
	}
	// 稀疏档
	for (const auto& item : info.xishudangList)
	{
		if (!item.isDraw)
		{
			auto& xishudangItem = processResult[item.index];
			config.text = QString("稀疏档 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xishudangItem, config);
		}
	}
	// 二维码
	for (const auto& item : info.erweimaList)
	{
		if (!item.isDraw)
		{
			auto& erweimaItem = processResult[item.index];
			config.text = QString("二维码 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, erweimaItem, config);
		}
	}
	// 大墨点
	for (const auto& item : info.damodianList)
	{
		if (!item.isDraw)
		{
			auto& damodianItem = processResult[item.index];
			config.text = QString("大墨点 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, damodianItem, config);
		}
	}
	// 孔洞
	for (const auto& item : info.kongdongList)
	{
		if (!item.isDraw)
		{
			auto& kongdongItem = processResult[item.index];
			config.text = QString("孔洞 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, kongdongItem, config);
		}
	}
	// 色标
	for (const auto& item : info.sebiaoList)
	{
		if (!item.isDraw)
		{
			auto& sebiaoItem = processResult[item.index];
			config.text = QString("色标 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, sebiaoItem, config);
		}
	}
	// 印刷缺陷
	for (const auto& item : info.yinshuaquexianList)
	{
		if (!item.isDraw)
		{
			auto& yinshuaquexianItem = processResult[item.index];
			config.text = QString("印刷缺陷 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, yinshuaquexianItem, config);
		}
	}
	// 小破洞
	for (const auto& item : info.xiaopodongList)
	{
		if (!item.isDraw)
		{
			auto& xiaopodongItem = processResult[item.index];
			config.text = QString("小破洞 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xiaopodongItem, config);
		}
	}
	// 胶带
	for (const auto& item : info.jiaodaiList)
	{
		if (!item.isDraw)
		{
			auto& jiaodaiItem = processResult[item.index];
			config.text = QString("胶带 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area * daichangxishuSqrt));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jiaodaiItem, config);
		}
	}
}

void ImageProcessorSmartCroppingOfBags::drawDefectRec_error(QImage& image,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex,
	const SmartCroppingOfBagsDefectInfo& info)
{
	auto& scoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	if (processResult.size() == 0)
	{
		return;
	}

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
	config.fontSize = 50;
	// 黑疤
	for (const auto& item : info.heibaList)
	{
		if (item.isDraw)
		{
			auto& heibaItem = processResult[item.index];
			config.text = QString("黑疤 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, heibaItem, config);
		}
	}
	// 疏档
	for (const auto& item : info.shudangList)
	{
		if (item.isDraw)
		{
			auto& shudangItem = processResult[item.index];
			config.text = QString("疏档 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, shudangItem, config);
		}
	}
	// 划破
	for (const auto& item : info.huapoList)
	{
		if (item.isDraw)
		{
			auto& huapoItem = processResult[item.index];
			config.text = QString("划破 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, huapoItem, config);
		}
	}
	// 接头
	for (const auto& item : info.jietouList)
	{
		if (item.isDraw)
		{
			auto& jietouItem = processResult[item.index];
			config.text = QString("接头 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jietouItem, config);
		}
	}
	// 挂丝
	for (const auto& item : info.guasiList)
	{
		if (item.isDraw)
		{
			auto& guasiItem = processResult[item.index];
			config.text = QString("挂丝 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, guasiItem, config);
		}
	}
	// 破洞
	for (const auto& item : info.podongList)
	{
		if (item.isDraw)
		{
			auto& podongItem = processResult[item.index];
			config.text = QString("破洞 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, podongItem, config);
		}
	}
	// 脏污
	for (const auto& item : info.zangwuList)
	{
		if (item.isDraw)
		{
			auto& zangwuItem = processResult[item.index];
			config.text = QString("脏污 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, zangwuItem, config);
		}
	}
	// 无疏档
	for (const auto& item : info.noshudangList)
	{
		if (item.isDraw)
		{
			auto& noshudangItem = processResult[item.index];
			config.text = QString("无疏档 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, noshudangItem, config);
		}
	}
	// 墨点
	for (const auto& item : info.modianList)
	{
		if (item.isDraw)
		{
			auto& modianItem = processResult[item.index];
			config.text = QString("墨点 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, modianItem, config);
		}
	}
	// 漏膜
	for (const auto& item : info.loumoList)
	{
		if (item.isDraw)
		{
			auto& loumoItem = processResult[item.index];
			config.text = QString("漏膜 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, loumoItem, config);
		}
	}
	// 稀疏档
	for (const auto& item : info.xishudangList)
	{
		if (item.isDraw)
		{
			auto& xishudangItem = processResult[item.index];
			config.text = QString("稀疏档 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xishudangItem, config);
		}
	}
	// 二维码
	for (const auto& item : info.erweimaList)
	{
		if (item.isDraw)
		{
			auto& erweimaItem = processResult[item.index];
			config.text = QString("二维码 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, erweimaItem, config);
		}
	}
	// 大墨点
	for (const auto& item : info.damodianList)
	{
		if (item.isDraw)
		{
			auto& damodianItem = processResult[item.index];
			config.text = QString("大墨点 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, damodianItem, config);
		}
	}
	// 孔洞
	for (const auto& item : info.kongdongList)
	{
		if (item.isDraw)
		{
			auto& kongdongItem = processResult[item.index];
			config.text = QString("孔洞 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, kongdongItem, config);
		}
	}
	// 色标
	for (const auto& item : info.sebiaoList)
	{
		if (item.isDraw)
		{
			auto& sebiaoItem = processResult[item.index];
			config.text = QString("色标 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, sebiaoItem, config);
		}
	}
	// 印刷缺陷
	for (const auto& item : info.yinshuaquexianList)
	{
		if (item.isDraw)
		{
			auto& yinshuaquexianItem = processResult[item.index];
			config.text = QString("印刷缺陷 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, yinshuaquexianItem, config);
		}
	}
	// 小破洞
	for (const auto& item : info.xiaopodongList)
	{
		if (item.isDraw)
		{
			auto& xiaopodongItem = processResult[item.index];
			config.text = QString("小破洞 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xiaopodongItem, config);
		}
	}
	// 胶带
	for (const auto& item : info.jiaodaiList)
	{
		if (item.isDraw)
		{
			auto& jiaodaiItem = processResult[item.index];
			config.text = QString("胶带 %1 %2").arg(static_cast<int>(item.score)).arg(static_cast<int>(item.area));
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jiaodaiItem, config);
		}
	}
}

void ImageProcessorSmartCroppingOfBags::setCollageImageNum(size_t num)
{
	_collageNum = num;
}


void ImageProcessingModuleSmartCroppingOfBags::BuildModule()
{
	_historyTimes = std::make_shared<rw::dsl::TimeBasedCache<Time, Time>>(50);
	_imageCollage = std::make_shared<ImageCollage>();
	_imageCollage->iniCache(50);
	_historyResult = std::make_shared<rw::dsl::TimeBasedCache<Time, HistoryDetectInfo>>(50);
	_timeBool = std::make_shared<rw::dsl::CacheFIFOThreadSafe<Time, bool>>(50);

	for (int i = 0; i < _numConsumers; ++i) {
		static size_t workIndexCount = 0;
		ImageProcessorSmartCroppingOfBags* processor = new ImageProcessorSmartCroppingOfBags(_queue, _mutex, _condition, workIndexCount, this);
		workIndexCount++;
		processor->buildSegModelEngine(modelEnginePath);
		processor->imageProcessingModuleIndex = index;
		connect(processor, &ImageProcessorSmartCroppingOfBags::imageReady, this, &ImageProcessingModuleSmartCroppingOfBags::imageReady, Qt::QueuedConnection);
		connect(processor, &ImageProcessorSmartCroppingOfBags::imageNGReady, this, &ImageProcessingModuleSmartCroppingOfBags::imageNGReady, Qt::QueuedConnection);
		_processors.push_back(processor);
		connect(processor, &ImageProcessorSmartCroppingOfBags::appendPixel, this, &ImageProcessingModuleSmartCroppingOfBags::appendPixel, Qt::QueuedConnection);

		processor->_historyResult = _historyResult;
		processor->_imageCollage = _imageCollage;
		processor->_historyTimes= _historyTimes;
		processor->_timeBool = _timeBool;
		processor->start();
	}

	mat1= cv::imread(R"(C:\Users\rw\Desktop\temp\2.jpg)");
	mat2 = cv::imread(R"(C:\Users\rw\Desktop\temp\2.jpg)");
}

void ImageProcessingModuleSmartCroppingOfBags::setCollageImageNum(size_t num)
{
	for (auto& item : _processors)
	{
		item->setCollageImageNum(num);
	}
}

ImageProcessingModuleSmartCroppingOfBags::ImageProcessingModuleSmartCroppingOfBags(int numConsumers, QObject* parent)
	: QObject(parent), _numConsumers(numConsumers) {

}

ImageProcessingModuleSmartCroppingOfBags::~ImageProcessingModuleSmartCroppingOfBags()
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

////监控io线程
//
//bool state = false;
//double lastlocation = 0;
//
////全局变量location
//double location = 0;
//void getlocation()
//{
//
//	auto& camera1 = GlobalStructDataSmartCroppingOfBags::getInstance().camera1;
//	//获得编码器的位置
//	while (true)
//	{
//		//location = camera1->getEncoderNumber();
//
//	}
//
//
//
//}








void ImageProcessingModuleSmartCroppingOfBags::onFrameCaptured(cv::Mat frame, size_t index)
{
	if (frame.empty()) {
		return; // 跳过空帧
	}

	auto& globalStructData = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& setConfig = globalStructData.setConfig;
	
	auto minFrameRows= setConfig.zuixiaochutu1* globalStructData.setConfig.daichangxishu1;
	auto maxFrameRows = setConfig.zuidachutu1 * globalStructData.setConfig.daichangxishu1;
	auto currentFrameRows = frame.rows;
	if (currentFrameRows<minFrameRows||currentFrameRows>maxFrameRows)
	{
		return;
	}

	Time currentTime = std::chrono::system_clock::now();
	rw::rqw::ElementInfo<cv::Mat> imagePart;
	frame.copyTo(imagePart.element );
	
	double nowLocation=0;
	globalStructData.camera1->getEncoderNumber(nowLocation);
	imagePart.attribute.insert("location", nowLocation);

	_historyTimes->insert(currentTime, currentTime);
	_timeBool->set(currentTime, false);


	//
	static bool temp{ false };


	if (temp)
	{
		imagePart.element = mat1.clone();
		temp = false;
	}
	else
	{
		imagePart.element = mat2.clone();
		temp = true;
	}



	//

	_imageCollage->pushImage(imagePart, currentTime);

	{
		QMutexLocker locker(&_mutex);
		MatInfo matInfo(imagePart);
		matInfo.location = nowLocation;
		matInfo.index = index;
		matInfo.time = currentTime;
		_queue.enqueue(matInfo);
		_condition.wakeOne();
	}
}

