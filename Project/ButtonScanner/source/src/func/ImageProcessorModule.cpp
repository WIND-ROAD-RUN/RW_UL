#include"stdafx.h"

#include"ime_utilty.hpp"

#include "GlobalStruct.h"
#include"ImageProcessorModule.h"
#include"ButtonUtilty.h"
#include"rqw_ImagePainter.h"

#include <QtConcurrent>

void ImageProcessor::buildModelEngineOT(const QString& enginePath)
{
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.5f;
	config.nms_threshold = 0.5f;
	config.modelPath = enginePath.toStdString();
	_modelEngineOT = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_obb, rw::ModelEngineDeployType::TensorRT);
}

//void ImageProcessor::buildModelEngineOnnxOO(const QString& enginePath, const QString& namePath)
//{
//	///*_modelEnginePtrOnnxOO.reset();*/
//	//_modelEnginePtrOnnxOO = std::make_unique<rw::imeoo::ModelEngineOO>(enginePath.toStdString(), namePath.toStdString());
//}
//
//void ImageProcessor::buildModelEngineOnnxSO(const QString& enginePath, const QString& namePath)
//{
//	/*_modelEnginePtrOnnxSO.reset();
//	_modelEnginePtrOnnxSO = std::make_unique<rw::imeso::ModelEngineSO>(enginePath.toStdString(), namePath.toStdString());*/
//}

std::vector<std::vector<size_t>> ImageProcessor::filterEffectiveIndexes_debug(std::vector<rw::DetectionRectangleInfo> info)
{
	auto processResultIndex = ImageProcessUtilty::getClassIndex(info);
	processResultIndex = getIndexInBoundary(info, processResultIndex);
	processResultIndex = ImageProcessUtilty::getAllIndexInMaxBody(info, processResultIndex);
	processResultIndex = getIndexInShieldingRange(info, processResultIndex);
	return processResultIndex;
}

std::vector<std::vector<size_t>> ImageProcessor::filterEffectiveIndexes_defect(
	std::vector<rw::DetectionRectangleInfo> info)
{
	auto& globalStruct = GlobalStructData::getInstance();

	auto processResultIndex = ImageProcessUtilty::getClassIndex(info);
	processResultIndex = getIndexInBoundary(info, processResultIndex);
	processResultIndex = ImageProcessUtilty::getAllIndexInMaxBody(info, processResultIndex);
	if (globalStruct.dlgProductSetConfig.shieldingRangeEnable)
	{
		processResultIndex = getIndexInShieldingRange(info, processResultIndex);
	}
	return processResultIndex;
}

//std::vector<rw::imeot::ProcessRectanglesResultOT> ImageProcessor::getDefectInBody(rw::imeot::ProcessRectanglesResultOT body, const std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResult)
//{
//	static int i = 0;
//	std::vector<rw::imeot::ProcessRectanglesResultOT> result;
//	auto& globalStruct = GlobalStructData::getInstance();
//
//	auto leleltMin = body.left_top.first;
//	auto leleltMax = body.right_bottom.first;
//	auto verticalMin = body.left_top.second;
//	auto verticalMax = body.right_bottom.second;
//
//	for (const auto& item : vecRecogResult)
//	{
//		if (leleltMin < item.center_x && item.center_x < leleltMax)
//		{
//			if (verticalMin < item.center_y && item.center_y < verticalMax)
//			{
//				result.emplace_back(item);
//			}
//		}
//	}
//	return result;
//}

//cv::Mat ImageProcessor::processAI(MatInfo& frame, QVector<QString>& errorInfo, std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResult, std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResultTarget)
//{
//	_isbad = false;
//	auto& globalStruct = GlobalStructData::getInstance();
//
//	cv::Mat resultImage1;
//	std::vector<rw::imeoo::ProcessRectanglesResultOO > vecReconResultOnnxOO;
//	std::vector<rw::imeso::ProcessRectanglesResultSO > vecReconResultOnnxSO;
//
//	auto openBladeShape = globalStruct.mainWindowConfig.isPositive && globalStruct.isOpenBladeShape;
//	auto openColor = globalStruct.mainWindowConfig.isPositive && globalStruct.isOpenColor;
//	auto openDefect = globalStruct.mainWindowConfig.isDefect;
//
//	QFuture<void> onnxFuture = QtConcurrent::run([&]() {
//		if (globalStruct.mainWindowConfig.isPositive)
//		{
//			if (openBladeShape)
//			{
//				_modelEnginePtrOnnxOO->ProcessMask(frame.image, resultImage1, vecReconResultOnnxOO);
//				LOG() vecReconResultOnnxOO.size();
//			}
//			if (openColor)
//			{
//				cv::Mat maskImage = cv::Mat::zeros(frame.image.size(), CV_8UC1);
//				_modelEnginePtrOnnxSO->ProcessMask(frame.image, resultImage1, maskImage, vecReconResultOnnxSO);
//			}
//		}
//		});
//
//	cv::Mat resultImage;
//	cv::Mat maskImage = cv::Mat::zeros(frame.image.size(), CV_8UC1);
//
//	if (openDefect)
//	{
//		_modelEnginePtr->ProcessMask(frame.image, resultImage, vecRecogResult);
//	}
//
//	onnxFuture.waitForFinished();
//
//	if (globalStruct.isOpenRemoveFunc || (globalStruct.isDebugMode)) {
//		bool hasBody = false;
//		auto body = getBody(vecRecogResult, hasBody);
//
//		bool hasBodyOnnxOO = false;
//		auto bodyOnnxOO = getBody(vecReconResultOnnxOO, hasBodyOnnxOO);
//
//		bool hasBodyOnnxSO = false;
//		auto bodyOnnxSO = getBody(vecReconResultOnnxSO, hasBodyOnnxSO);
//
//		if (!openBladeShape)
//		{
//			hasBodyOnnxOO = true;
//		}
//		if (!openColor)
//		{
//			hasBodyOnnxSO = true;
//		}
//		if (!openDefect)
//		{
//			hasBody = true;
//		}
//
//
//		LOG() "" << "hasBodyOnnxOO:" << hasBodyOnnxOO << "hasBodyOnnxSO:" << hasBodyOnnxSO << "hasBody:" << hasBody;
//
//		if ((!hasBody) || (!hasBodyOnnxOO) || (!hasBodyOnnxSO))
//		{
//			if (globalStruct.isOpenRemoveFunc) {
//				globalStruct.statisticalInfo.wasteCount++;
//
//				if (imageProcessingModuleIndex == 2 || imageProcessingModuleIndex == 4) {
//					globalStruct.statisticalInfo.produceCount++;
//				}
//
//				float absLocation = frame.location;
//				if (absLocation < 0) {
//					absLocation = -absLocation; // 将负值转换为正值
//				}
//
//				switch (imageProcessingModuleIndex)
//				{
//				case 1:
//					globalStruct.productPriorityQueue1.push(absLocation);
//					break;
//				case 2:
//					globalStruct.productPriorityQueue2.push(absLocation);
//					break;
//				case 3:
//					globalStruct.productPriorityQueue3.push(absLocation);
//					break;
//				case 4:
//					globalStruct.productPriorityQueue4.push(absLocation);
//					break;
//				default:
//					break;
//				}
//			}
//			_isbad = true;
//			_isbad = true;
//			if (globalStruct.isTakePictures) {
//				globalStruct.imageSaveEngine->pushImage(cvMatToQImage(frame.image), "NG", "Button");
//				globalStruct.imageSaveEngine->pushImage(cvMatToQImage(frame.image), "OK", "Button");
//			}
//		}
//		else
//		{
//			if (openDefect || openColor || openBladeShape)
//			{
//				auto defect = getDefectInBody(body, vecRecogResult);
//				eliminationLogic(frame,
//					frame.image,
//					errorInfo,
//					defect,
//					vecRecogResultTarget,
//					vecReconResultOnnxOO,
//					vecReconResultOnnxSO);
//			}
//		}
//	}
//
//	return frame.image.clone();
//}

//rw::imeot::ProcessRectanglesResultOT ImageProcessor::getBody(std::vector<rw::imeot::ProcessRectanglesResultOT>& processRectanglesResult, bool& hasBody)
//{
//	hasBody = false;
//	rw::imeot::ProcessRectanglesResultOT result;
//	result.width = 0;
//	result.height = 0;
//	for (auto& i : processRectanglesResult)
//	{
//		if (i.classID == 0)
//		{
//			auto isIn = isInArea(i.center_x);
//			if (isIn)
//			{
//				if ((i.width * i.height) > (result.width * result.height))
//				{
//					result = i;
//					hasBody = true;
//				}
//			}
//		}
//	}
//	return result;
//}

//rw::imeoo::ProcessRectanglesResultOO ImageProcessor::getBody(
//	std::vector<rw::imeoo::ProcessRectanglesResultOO>& processRectanglesResult, bool& hasBody)
//{
//	hasBody = false;
//	rw::imeoo::ProcessRectanglesResultOO result;
//	int area = 0;
//	LOG() processRectanglesResult.size();
//	for (auto& i : processRectanglesResult)
//	{
//		if (i.classID == 0)
//		{
//			auto center_x = i.left_top.first + (i.right_bottom.first - i.left_top.first) / 2;
//			auto center_y = i.left_top.second + (i.right_bottom.second - i.left_top.second) / 2;
//			auto width = i.right_bottom.first - i.left_top.first;
//			auto height = i.right_bottom.second - i.left_top.second;
//			auto isIn = isInArea(center_x);
//			if (isIn)
//			{
//				if ((width * height) > area)
//				{
//					result = i;
//					hasBody = true;
//					area = width * height;
//				}
//			}
//		}
//	}
//	return result;
//}

//rw::imeso::ProcessRectanglesResultSO ImageProcessor::getBody(
//	std::vector<rw::imeso::ProcessRectanglesResultSO>& processRectanglesResult, bool& hasBody)
//{
//	hasBody = false;
//	rw::imeso::ProcessRectanglesResultSO result;
//	int area = 0;
//	for (auto& i : processRectanglesResult)
//	{
//		if (i.classID == 0)
//		{
//			auto center_x = i.left_top.first + (i.right_bottom.first - i.left_top.first) / 2;
//			auto center_y = i.left_top.second + (i.right_bottom.second - i.left_top.second) / 2;
//			auto width = i.right_bottom.first - i.left_top.first;
//			auto height = i.right_bottom.second - i.left_top.second;
//			auto isIn = isInArea(center_x);
//			if (isIn)
//			{
//				if ((width * height) > area)
//				{
//					result = i;
//					hasBody = true;
//					area = width * height;
//				}
//			}
//		}
//	}
//	return result;
//}

//void
//ImageProcessor::eliminationLogic(
//	MatInfo& frame,
//	cv::Mat& resultImage, QVector<QString>& errorInfo,
//	std::vector<rw::imeot::ProcessRectanglesResultOT>& processRectanglesResult,
//	std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResultTarget,
//	std::vector<rw::imeoo::ProcessRectanglesResultOO>& processRectanglesResultOO, std::vector<rw::imeso::ProcessRectanglesResultSO>& processRectanglesResultSO)
//{
//	auto saveIamge = resultImage.clone();
//	auto& globalStruct = GlobalStructData::getInstance();
//
//	auto& systemConfig = globalStruct.dlgProduceLineSetConfig;
//	auto& checkConfig = globalStruct.dlgProductSetConfig;
//	auto& mainWindowSet = globalStruct.mainWindowConfig;
//	auto& dlgHideScoreSet = globalStruct.dlgHideScoreSetConfig;
//	auto openDefect = globalStruct.mainWindowConfig.isDefect;
//
//	double& pixEquivalent = systemConfig.pixelEquivalent1;
//	switch (frame.index)
//	{
//	case 2:
//		pixEquivalent = systemConfig.pixelEquivalent2;
//		break;
//	case 3:
//		pixEquivalent = systemConfig.pixelEquivalent3;
//		break;
//	case 4:
//		pixEquivalent = systemConfig.pixelEquivalent4;
//		break;
//	default:
//		break;
//	}
//
//	auto isBad = false;
//
//	cv::Mat resultMat;
//	cv::Mat maskmat;
//
//	std::vector<int> waiJingIndexs = std::vector<int>();
//	size_t holesCount = 0;
//	std::vector<int> konJingIndexs = std::vector<int>();
//	std::vector<int> daPoBianIndexs = std::vector<int>();
//	std::vector<int> qiKonIndexs = std::vector<int>();
//	std::vector<int> duYanIndexs = std::vector<int>();
//	std::vector<int> moShiIndexs = std::vector<int>();
//	std::vector<int> liaoTouIndexs = std::vector<int>();
//	std::vector<int> youQiIndexs = std::vector<int>();
//	std::vector<int> lieHenIndexs = std::vector<int>();
//	std::vector<int> poYanIndexs = std::vector<int>();
//
//	for (int i = 0; i < processRectanglesResult.size(); i++)
//	{
//		switch (processRectanglesResult[i].classID)
//		{
//		case 0: waiJingIndexs.push_back(i); continue;
//		case 1: holesCount++; konJingIndexs.push_back(i); continue;
//		case 2: daPoBianIndexs.push_back(i); continue;
//		case 3: qiKonIndexs.push_back(i); continue;
//		case 4: duYanIndexs.push_back(i); continue;
//		case 5: moShiIndexs.push_back(i); continue;
//		case 6: liaoTouIndexs.push_back(i); continue;
//		case 7: youQiIndexs.push_back(i); continue;
//		case 8: lieHenIndexs.push_back(i); continue;
//		case 9: poYanIndexs.push_back(i); continue;
//
//		default: continue;
//		}
//	}
//
//	// 检查外径是否被识别成小孔
//	if (!konJingIndexs.empty()) {
//		// 计算小孔的平均高度和宽度
//		double totalHeight = 0.0, totalWidth = 0.0;
//		for (int index : konJingIndexs) {
//			const auto& rect = processRectanglesResult[index];
//			int height = rect.right_bottom.second - rect.left_top.second;
//			int width = rect.right_bottom.first - rect.left_top.first;
//			totalHeight += height;
//			totalWidth += width;
//		}
//		double avgHeight = totalHeight / konJingIndexs.size();
//		double avgWidth = totalWidth / konJingIndexs.size();
//
//		// 允许的偏差比例（可以根据实际情况调整）
//		const double deviationRatio = 0.8;
//
//		// 剔除偏离平均值过大的矩形
//		for (int i = 0; i < konJingIndexs.size(); ++i) {
//			int index = konJingIndexs[i];
//			const auto& rect = processRectanglesResult[index];
//			int height = rect.right_bottom.second - rect.left_top.second;
//			int width = rect.right_bottom.first - rect.left_top.first;
//
//			// 判断是否偏离平均值
//			if (std::abs(height - avgHeight) > avgHeight * deviationRatio ||
//				std::abs(width - avgWidth) > avgWidth * deviationRatio) {
//				// 偏离过大，剔除
//				konJingIndexs.erase(konJingIndexs.begin() + i);
//				holesCount--;
//				--i; // 调整索引以避免跳过下一个元素
//			}
//		}
//	}
//
//	std::vector<rw::imeot::ProcessRectanglesResultOT> body;
//	std::vector<rw::imeot::ProcessRectanglesResultOT> hole;
//
//	//拾取外径和孔径
//	for (int i = 0;i < waiJingIndexs.size();i++)
//	{
//		body.emplace_back(processRectanglesResult[waiJingIndexs[i]]);
//	}
//	for (int i = 0;i < konJingIndexs.size();i++)
//	{
//		hole.emplace_back(processRectanglesResult[konJingIndexs[i]]);
//	}
//
//	if (mainWindowSet.isPositive)
//	{
//		if (globalStruct.isOpenBladeShape)
//		{
//			if (processRectanglesResultOO.size() != 0)
//			{
//				if ((processRectanglesResultOO.at(0).classID == 0) && (processRectanglesResultOO.at(0).score > dlgHideScoreSet.forAndAgainstScore))
//				{
//					errorInfo.emplace_back("刀型错误");
//					isBad = true;
//					_isbad = true;
//				}
//			}
//		}
//		if (globalStruct.isOpenColor)
//		{
//			if (processRectanglesResultSO.size() != 0)
//			{
//				if ((processRectanglesResultSO.at(0).classID == 0) && (processRectanglesResultSO.at(0).score > dlgHideScoreSet.forAndAgainstScore))
//				{
//					errorInfo.emplace_back("颜色错误");
//					isBad = true;
//					_isbad = true;
//				}
//			}
//		}
//	}
//
//	//检查色差
//	if (checkConfig.specifyColorDifferenceEnable && openDefect)
//	{
//		int RMin = checkConfig.specifyColorDifferenceR-checkConfig.specifyColorDifferenceDeviation;
//		RMin = std::clamp(RMin, 0, 255);
//		int RMax = checkConfig.specifyColorDifferenceR - checkConfig.specifyColorDifferenceDeviation;
//		RMax = std::clamp(RMax, 0, 255);
//		int GMin = checkConfig.specifyColorDifferenceG - checkConfig.specifyColorDifferenceDeviation;
//		GMin = std::clamp(GMin, 0, 255);
//		int GMax = checkConfig.specifyColorDifferenceG + checkConfig.specifyColorDifferenceDeviation;
//		GMax = std::clamp(GMax, 0, 255);
//		int BMin = checkConfig.specifyColorDifferenceB - checkConfig.specifyColorDifferenceDeviation;
//		BMin = std::clamp(BMin, 0, 255);
//		int BMax = checkConfig.specifyColorDifferenceB + checkConfig.specifyColorDifferenceDeviation;
//		BMax = std::clamp(BMax, 0, 255);
//
//		if (!body.empty())
//		{
//			auto top_left_x = body[0].center_x - (body[0].width / 2);
//			auto top_left_y = body[0].center_y - (body[0].height / 2);
//			cv::Rect rect(top_left_x, top_left_y, body[0].width, body[0].height);
//
//			std::vector<cv::Rect> excludeRegions;
//			for (int i = 0; i < konJingIndexs.size(); i++)
//			{
//				auto top_left_x = processRectanglesResult[konJingIndexs[i]].center_x - (processRectanglesResult[konJingIndexs[i]].width / 2);
//				auto top_left_y = processRectanglesResult[konJingIndexs[i]].center_y - (processRectanglesResult[konJingIndexs[i]].height / 2);
//				cv::Rect excludeRect(top_left_x, top_left_y, processRectanglesResult[konJingIndexs[i]].width, processRectanglesResult[konJingIndexs[i]].height);
//				excludeRegions.push_back(excludeRect);
//			}
//
//			auto currentRGB = 
//				ImageProcessUtilty::calculateRegionRGB(frame.image, rect, 
//					ImageProcessUtilty::CropMode::InscribedCircle, excludeRegions,ImageProcessUtilty::CropMode::InscribedCircle);
//
//			if (RMin <=currentRGB[0]&& currentRGB[0]<=RMax)
//			{
//				isBad = true;
//				_isbad = true;
//			}
//			else
//			{
//				isBad = false;
//				_isbad = false;
//				errorInfo.emplace_back("B数值不在指定范围内");
//			}
//
//			if (GMin <= currentRGB[1] && currentRGB[1] <= GMax)
//			{
//				isBad = true;
//				_isbad = true;
//			}
//			else
//			{
//				isBad = false;
//				_isbad = false;
//				errorInfo.emplace_back("G数值不在指定范围内");
//			}
//
//			if (BMin <= currentRGB[2] && currentRGB[2] <= BMax)
//			{
//				isBad = true;
//				_isbad = true;
//			}
//			else
//			{
//				isBad = false;
//				_isbad = false;
//				errorInfo.emplace_back("B数值不在指定范围内");
//			}
//		}
//	}
//
//	//检查大色差
//	if (checkConfig.largeColorDifferenceEnable && openDefect)
//	{
//		//添加刷新逻辑防止下一次启动的时候无法筛选
//		static std::vector<int> RList;
//		static std::vector<int> GList;
//		static std::vector<int> BList;
//		if (!body.empty())
//		{
//			auto top_left_x = body[0].center_x - (body[0].width / 2);
//			auto top_left_y = body[0].center_y - (body[0].height / 2);
//			cv::Rect rect(top_left_x, top_left_y, body[0].width, body[0].height);
//
//			std::vector<cv::Rect> excludeRegions;
//			for (int i = 0; i < konJingIndexs.size(); i++)
//			{
//				auto top_left_x = processRectanglesResult[konJingIndexs[i]].center_x - (processRectanglesResult[konJingIndexs[i]].width / 2);
//				auto top_left_y = processRectanglesResult[konJingIndexs[i]].center_y - (processRectanglesResult[konJingIndexs[i]].height / 2);
//				cv::Rect excludeRect(top_left_x, top_left_y, processRectanglesResult[konJingIndexs[i]].width, processRectanglesResult[konJingIndexs[i]].height);
//				excludeRegions.push_back(excludeRect);
//			}
//
//			auto currentRGB =
//				ImageProcessUtilty::calculateRegionRGB(frame.image, rect,
//					ImageProcessUtilty::CropMode::InscribedCircle, excludeRegions, ImageProcessUtilty::CropMode::InscribedCircle);
//
//			if (RList.size() < 3)
//			{
//				if (RList.size() < 3)
//				{
//					RList.push_back(currentRGB[0]);
//				}
//
//				if (GList.size() < 3)
//				{
//					GList.push_back(currentRGB[1]);
//				}
//
//				if (BList.size() < 3)
//				{
//					BList.push_back(currentRGB[2]);
//				}
//			}
//			else
//			{
//				int averageR = static_cast<int>(std::accumulate(RList.begin(), RList.end(), 0)) / RList.size();
//				int averageG = static_cast<int>(std::accumulate(GList.begin(), GList.end(), 0)) / GList.size();
//				int averageB = static_cast<int>(std::accumulate(BList.begin(), BList.end(), 0)) / BList.size();
//
//				int RMin = averageR - checkConfig.largeColorDifferenceDeviation;
//				RMin = std::clamp(RMin, 0, 255);
//				int RMax = averageR - checkConfig.largeColorDifferenceDeviation;
//				RMax = std::clamp(RMax, 0, 255);
//				int GMin = averageG - checkConfig.largeColorDifferenceDeviation;
//				GMin = std::clamp(GMin, 0, 255);
//				int GMax = averageG + checkConfig.largeColorDifferenceDeviation;
//				GMax = std::clamp(GMax, 0, 255);
//				int BMin = averageB - checkConfig.largeColorDifferenceDeviation;
//				BMin = std::clamp(BMin, 0, 255);
//				int BMax = averageB + checkConfig.largeColorDifferenceDeviation;
//				BMax = std::clamp(BMax, 0, 255);
//
//				if (RMin <= currentRGB[0] && currentRGB[0] <= RMax)
//				{
//					isBad = true;
//					_isbad = true;
//				}
//				else
//				{
//					isBad = false;
//					_isbad = false;
//					errorInfo.emplace_back("B数值不在指定范围内");
//				}
//
//				if (GMin <= currentRGB[1] && currentRGB[1] <= GMax)
//				{
//					isBad = true;
//					_isbad = true;
//				}
//				else
//				{
//					isBad = false;
//					_isbad = false;
//					errorInfo.emplace_back("G数值不在指定范围内");
//				}
//
//				if (BMin <= currentRGB[2] && currentRGB[2] <= BMax)
//				{
//					isBad = true;
//					_isbad = true;
//				}
//				else
//				{
//					isBad = false;
//					_isbad = false;
//					errorInfo.emplace_back("B数值不在指定范围内");
//				}
//
//			}
//
//		}
//	}
//
//	//检查外径
//	if (checkConfig.outsideDiameterEnable && openDefect)
//	{
//		ImagePainter::drawCirclesOnImage(resultImage, body);
//		if (waiJingIndexs.size() == 0)
//		{
//			isBad = true;
//			_isbad = true;
//			errorInfo.emplace_back("没找到外径");
//		}
//		else
//		{
//			auto shangXiaPianCha = processRectanglesResult[waiJingIndexs[0]].right_bottom.second - processRectanglesResult[waiJingIndexs[0]].left_top.second - checkConfig.outsideDiameterValue / pixEquivalent;
//			auto zuoYouPianCha = processRectanglesResult[waiJingIndexs[0]].right_bottom.first - processRectanglesResult[waiJingIndexs[0]].left_top.first - checkConfig.outsideDiameterValue / pixEquivalent;
//
//			auto shangXiaPianChaAbs = abs(shangXiaPianCha);
//			auto zuoYouPianChaAbs = abs(zuoYouPianCha);
//
//			if (shangXiaPianChaAbs > checkConfig.outsideDiameterDeviation / pixEquivalent || zuoYouPianChaAbs > checkConfig.outsideDiameterDeviation / pixEquivalent)
//			{
//				isBad = true;
//				_isbad = true;
//
//				if (shangXiaPianChaAbs >= zuoYouPianChaAbs)
//				{
//					errorInfo.emplace_back("外径 " + QString::number(shangXiaPianCha * pixEquivalent));
//				}
//				else
//				{
//					errorInfo.emplace_back("外径 " + QString::number(zuoYouPianCha * pixEquivalent));
//				}
//			}
//		}
//	}
//
//	//检查孔数
//	if (checkConfig.holesCountEnable && openDefect)
//	{
//		ImagePainter::drawCirclesOnImage(resultImage, hole);
//		// 获取当前时间
//		auto now = std::chrono::system_clock::now();
//		auto duration = now.time_since_epoch();
//		auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
//		auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
//		auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
//
//		// 格式化时间
//		std::ostringstream timeStream;
//		timeStream << std::setfill('0') << std::setw(2) << minutes << "_"
//			<< std::setfill('0') << std::setw(2) << seconds << "_"
//			<< std::setfill('0') << std::setw(3) << millis;
//
//		if (holesCount != checkConfig.holesCountValue)
//		{
//			isBad = true;
//			_isbad = true;
//			errorInfo.emplace_back("只找到" + QString::number(holesCount) + "个孔");
//		}
//	}
//
//	//检查大破边
//	if (checkConfig.edgeDamageEnable && openDefect)
//	{
//		for (int i = 0; i < daPoBianIndexs.size(); i++)
//		{
//			auto score = processRectanglesResult[daPoBianIndexs[i]].score;
//
//			if (score >= (checkConfig.edgeDamageSimilarity) / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("破边 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < daPoBianIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[daPoBianIndexs[i]]);
//				}
//			}
//		}
//	}
//	//检查气孔
//	if (checkConfig.poreEnable && openDefect)
//	{
//		for (int i = 0; i < qiKonIndexs.size(); i++)
//		{
//			auto score = processRectanglesResult[qiKonIndexs[i]].score;
//			if (score >= checkConfig.poreEnableScore / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("气孔 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < qiKonIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[qiKonIndexs[i]]);
//				}
//			}
//		}
//	}
//
//	//检查堵眼
//	if (checkConfig.blockEyeEnable && openDefect)
//	{
//		for (int i = 0; i < duYanIndexs.size(); i++)
//		{
//			auto score = processRectanglesResult[duYanIndexs[i]].score;
//			if (score >= checkConfig.blockEyeEnableScore / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("堵眼 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < duYanIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[duYanIndexs[i]]);
//				}
//			}
//		}
//	}
//
//	//检查磨石
//	if (checkConfig.grindStoneEnable && openDefect)
//	{
//		for (int i = 0; i < moShiIndexs.size(); i++)
//		{
//			auto score = processRectanglesResult[moShiIndexs[i]].score;
//			if (score >= checkConfig.grindStoneEnableScore / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("磨石 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < moShiIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[moShiIndexs[i]]);
//				}
//			}
//		}
//	}
//
//	//检查料头
//	if (checkConfig.materialHeadEnable && openDefect)
//	{
//		for (int i = 0; i < liaoTouIndexs.size(); i++)
//		{
//			auto score = processRectanglesResult[liaoTouIndexs[i]].score;
//			if (score >= checkConfig.materialHeadEnableScore / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("料头 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < liaoTouIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[liaoTouIndexs[i]]);
//				}
//			}
//		}
//	}
//
//	//检查脏污
//	if (checkConfig.paintEnable && openDefect)
//	{
//		for (int i = 0; i < youQiIndexs.size(); i++)
//		{
//			auto score = processRectanglesResult[youQiIndexs[i]].score;
//			if (score >= checkConfig.paintEnableScore / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("油漆 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < youQiIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[youQiIndexs[i]]);
//				}
//			}
//		}
//	}
//
//	//检查裂痕
//	if (checkConfig.crackEnable && openDefect)
//	{
//		for (int i = 0; i < lieHenIndexs.size(); i++)
//		{
//			auto width = abs(processRectanglesResult[lieHenIndexs[i]].right_bottom.first - processRectanglesResult[lieHenIndexs[i]].left_top.first);
//			auto height = abs(processRectanglesResult[lieHenIndexs[i]].right_bottom.second - processRectanglesResult[lieHenIndexs[i]].left_top.second);
//
//			auto score = processRectanglesResult[lieHenIndexs[i]].score;
//			if (score >= checkConfig.crackSimilarity / 100)
//			{
//				isBad = true;
//				_isbad = true;
//				errorInfo.emplace_back("裂痕 " + QString::number(static_cast<int>(score * 100)));
//				for (int i = 0;i < lieHenIndexs.size();i++)
//				{
//					vecRecogResultTarget.emplace_back(processRectanglesResult[lieHenIndexs[i]]);
//				}
//			}
//		}
//	}
//
//	////检查小气孔
//
//	//if (checkConfig.brokenEyeEnable)
//	//{
//	//	for (int i = 0; i < poYanIndexs.size(); i++)
//	//	{
//	//		auto score = processRectanglesResult[poYanIndexs[i]].score;
//	//		auto width = abs(processRectanglesResult[poYanIndexs[i]].right_bottom.first - processRectanglesResult[poYanIndexs[i]].left_top.first);
//	//		auto height = abs(processRectanglesResult[poYanIndexs[i]].right_bottom.second - processRectanglesResult[poYanIndexs[i]].left_top.second);
//	//		if (score >= checkConfig.brokenEyeSimilarity)
//	//		{
//	//			isBad = true;
//	//			errorInfo.emplace_back("破眼 " + QString::number(score));
//	//			for (int i = 0;i < poYanIndexs.size();i++)
//	//			{
//	//				vecRecogResultTarget.emplace_back(processRectanglesResult[poYanIndexs[i]]);
//	//			}
//	//		}
//	//	}
//	//}
//
//	////检查小破边
//	//if (checkConfig.apertureEnable)
//	//{
//	//	for (int i = 0; i < konJingIndexs.size(); i++)
//	//	{
//	//		auto shangXiaPianCha = processRectanglesResult[konJingIndexs[i]].right_bottom.second - processRectanglesResult[konJingIndexs[i]].left_top.second - checkConfig.apertureValue / pixEquivalent;
//	//		auto zuoYouPianCha = processRectanglesResult[konJingIndexs[i]].right_bottom.first - processRectanglesResult[konJingIndexs[i]].left_top.first - checkConfig.apertureValue / pixEquivalent;
//
//	//		auto shangXiaPianChaAbs = abs(shangXiaPianCha);
//	//		auto zuoYouPianChaAbs = abs(zuoYouPianCha);
//
//	//		if (shangXiaPianChaAbs > checkConfig.apertureSimilarity / pixEquivalent || zuoYouPianChaAbs > checkConfig.apertureSimilarity / pixEquivalent)
//	//		{
//	//			isBad = true;
//
//	//			if (shangXiaPianChaAbs >= zuoYouPianChaAbs)
//	//			{
//	//				errorInfo.emplace_back("孔径 " + QString::number(shangXiaPianCha * pixEquivalent));
//	//				for (int i = 0;i < poYanIndexs.size();i++)
//	//				{
//	//					vecRecogResultTarget.emplace_back(processRectanglesResult[poYanIndexs[i]]);
//	//				}
//	//			}
//	//			else
//	//			{
//	//				errorInfo.emplace_back("孔径 " + QString::number(zuoYouPianCha * pixEquivalent));
//	//				for (int i = 0;i < poYanIndexs.size();i++)
//	//				{
//	//					vecRecogResultTarget.emplace_back(processRectanglesResult[poYanIndexs[i]]);
//	//				}
//	//			}
//
//	//		}
//	//	}
//	//}
//
//	//检查孔心距
//	if (checkConfig.holeCenterDistanceEnable && openDefect)
//	{
//		for (int i = 0; i < konJingIndexs.size(); i++)
//		{
//			auto konCenterY = processRectanglesResult[konJingIndexs[i]].left_top.second + (processRectanglesResult[konJingIndexs[i]].right_bottom.second - processRectanglesResult[konJingIndexs[i]].left_top.second) / 2;
//			auto konCenterX = processRectanglesResult[konJingIndexs[i]].left_top.first + (processRectanglesResult[konJingIndexs[i]].right_bottom.first - processRectanglesResult[konJingIndexs[i]].left_top.first) / 2;
//
//			auto konXinJu = std::sqrt((konCenterX * frame.image.cols / 2) + (konCenterY * frame.image.rows / 2));
//			auto pianCha = konXinJu - checkConfig.holeCenterDistanceValue / pixEquivalent;
//
//			if (abs(pianCha) > checkConfig.holeCenterDistanceSimilarity / pixEquivalent)
//			{
//				isBad = true;
//				_isbad = true;
//
//				errorInfo.emplace_back("孔心距 " + QString::number(pianCha * pixEquivalent));
//			}
//		}
//	}
//
//	if (globalStruct.isOpenRemoveFunc) {
//		if (isBad) {
//			globalStruct.statisticalInfo.wasteCount++;
//		}
//
//		if (imageProcessingModuleIndex == 2 || imageProcessingModuleIndex == 4) {
//			globalStruct.statisticalInfo.produceCount++;
//		}
//
//		if (isBad) {
//			float absLocation = frame.location;
//			if (absLocation < 0) {
//				absLocation = -absLocation; // 将负值转换为正值
//			}
//
//			switch (imageProcessingModuleIndex)
//			{
//			case 1:
//				globalStruct.productPriorityQueue1.push(absLocation);
//				break;
//			case 2:
//				globalStruct.productPriorityQueue2.push(absLocation);
//				break;
//			case 3:
//				globalStruct.productPriorityQueue3.push(absLocation);
//				break;
//			case 4:
//				globalStruct.productPriorityQueue4.push(absLocation);
//				break;
//			default:
//				break;
//			}
//		}
//	}
//
//	if (globalStruct.isTakePictures) {
//		if (isBad) {
//			globalStruct.imageSaveEngine->pushImage(cvMatToQImage(saveIamge), "NG", "Button");
//		}
//		else {
//			globalStruct.imageSaveEngine->pushImage(cvMatToQImage(saveIamge), "OK", "Button");
//		}
//	}
//}

//void ImageProcessor::drawErrorLocate(QImage& image, std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResult, const QColor& drawColor)
//{
//	auto& globalStructLineSetConfig = GlobalStructData::getInstance().dlgProduceLineSetConfig;
//	double pixEquivalent;
//	switch (imageProcessingModuleIndex)
//	{
//	case 1:
//		pixEquivalent = globalStructLineSetConfig.pixelEquivalent1;
//		break;
//	case 2:
//		pixEquivalent = globalStructLineSetConfig.pixelEquivalent2;
//		break;
//	case 3:
//		pixEquivalent = globalStructLineSetConfig.pixelEquivalent3;
//		break;
//	case 4:
//		pixEquivalent = globalStructLineSetConfig.pixelEquivalent4;
//		break;
//	}
//
//	auto& checkConfig = GlobalStructData::getInstance().dlgProductSetConfig;
//	if (image.isNull()) {
//		return;
//	}
//	for (const auto& item : vecRecogResult) {
//		if (item.classID == 0 || item.classID == 1) {
//			continue;
//		}
//		if (!checkConfig.edgeDamageEnable && item.classID == 2)
//		{
//			continue;
//		}
//		if (!checkConfig.poreEnable && item.classID == 3)
//		{
//			continue;
//		}
//		if (!checkConfig.blockEyeEnable && item.classID == 4)
//		{
//			continue;
//		}
//		if (!checkConfig.grindStoneEnable && item.classID == 5)
//		{
//			continue;
//		}
//		if (!checkConfig.materialHeadEnable && item.classID == 6)
//		{
//			continue;
//		}
//		if (!checkConfig.paintEnable && item.classID == 7)
//		{
//			continue;
//		}
//		if (!checkConfig.crackEnable && item.classID == 8)
//		{
//			continue;
//		}
//		if (!checkConfig.brokenEyeEnable && item.classID == 9)
//		{
//			continue;
//		}
//
//		auto leftTop = item.left_top;
//		auto rightBottom = item.right_bottom;
//
//		// 绘制矩形框
//		QPainter painter(&image);
//		painter.setPen(QPen(drawColor, 5)); // 使用传入的颜色
//		painter.drawRect(QRect(leftTop.first, leftTop.second, rightBottom.first - leftTop.first, rightBottom.second - leftTop.second));
//
//		// 设置字体大小
//		QFont font = painter.font();
//		font.setPixelSize(50); // 将字体大小设置为 50 像素（可以根据需要调整）
//		painter.setFont(font);
//
//		// 绘制文字
//		QString text;
//
//		switch (item.classID)
//		{
//		case 2:
//			text = "破边";
//			break;
//		case 3:
//			text = "气孔";
//			break;
//		case 4:
//			text = "堵眼";
//			break;
//		case 5:
//			text = "磨石";
//			break;
//		case 6:
//			text = "料头";
//			break;
//		case 7:
//			text = "脏污";
//			break;
//		case 8:
//			text = "裂痕";
//			break;
//		case 9:
//			text = "破眼";
//			break;
//		case 10:
//			text = "小气孔";
//			break;
//		case 11:
//			text = "毛发";
//			break;
//		case 12:
//			text = "小破边";
//			break;
//		case 13:
//			text = "白边";
//			break;
//		default:
//			text = QString::number(item.classID);
//			break;
//		}
//		int score = item.score * 100;
//		auto area = std::round(item.height * item.width * pixEquivalent * 10) / 10.0; // 保留一位小数
//		text = text + QString::number(score) + " " + QString::number(area);
//
//		// 设置文字颜色
//		painter.setPen(drawColor); // 使用传入的颜色
//		painter.drawText(leftTop.first, leftTop.second - 5, text);
//	}
//}

void ImageProcessor::drawLine(QImage& image)
{
	auto& index = imageProcessingModuleIndex;
	auto& dlgProduceLineSetConfig = GlobalStructData::getInstance().dlgProduceLineSetConfig;
	auto& checkConfig = GlobalStructData::getInstance().dlgProductSetConfig;
	if (index == 1)
	{
		drawLine_locate(image, dlgProduceLineSetConfig.limit1);
		drawLine_locate(image, dlgProduceLineSetConfig.limit1 + (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent1));
	}
	else if (index == 2)
	{
		drawLine_locate(image, dlgProduceLineSetConfig.limit2);
		drawLine_locate(image, dlgProduceLineSetConfig.limit2 - (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent2));
	}
	else if (index == 3)
	{
		drawLine_locate(image, dlgProduceLineSetConfig.limit3);
		drawLine_locate(image, dlgProduceLineSetConfig.limit3 + (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent3));
	}
	else if (index == 4)
	{
		drawLine_locate(image, dlgProduceLineSetConfig.limit4);
		drawLine_locate(image, dlgProduceLineSetConfig.limit4 - (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent4));
	}
}

void ImageProcessor::drawLine_locate(QImage& image, size_t locate)
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

void ImageProcessor::drawVerticalBoundaryLine(QImage& image)
{
	auto& index = imageProcessingModuleIndex;
	auto& dlgProduceLineSetConfig = GlobalStructData::getInstance().dlgProduceLineSetConfig;
	auto& checkConfig = GlobalStructData::getInstance().dlgProductSetConfig;
	rw::rqw::ImagePainter::PainterConfig painterConfig;
	painterConfig.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Orange);
	if (index == 1)
	{
		auto limit1_1 = dlgProduceLineSetConfig.limit1;
		auto limit1_2 = dlgProduceLineSetConfig.limit1 + (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent1);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit1_1, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit1_2, painterConfig);
	}
	else if (index == 2)
	{
		auto limit2_1 = dlgProduceLineSetConfig.limit2;
		auto limit2_2 = dlgProduceLineSetConfig.limit2 - (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent2);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit2_1, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit2_2, painterConfig);
	}
	else if (index == 3)
	{
		auto limit3_1 = dlgProduceLineSetConfig.limit3;
		auto limit3_2 = dlgProduceLineSetConfig.limit3 + (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent3);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit3_1, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit3_2, painterConfig);
	}
	else if (index == 4)
	{
		auto limit4_1 = dlgProduceLineSetConfig.limit4;
		auto limit4_2 = dlgProduceLineSetConfig.limit4 - (checkConfig.outsideDiameterValue / dlgProduceLineSetConfig.pixelEquivalent4);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit4_1, painterConfig);
		rw::rqw::ImagePainter::drawVerticalLine(image, limit4_2, painterConfig);
	}
}

void ImageProcessor::drawButtonDefectInfoText(QImage& image, const ButtonDefectInfo& info)
{
	QVector<QString> textList;
	std::vector<rw::rqw::ImagePainter::PainterConfig> configList;
	rw::rqw::ImagePainter::PainterConfig config;

	//运行时间
	configList.push_back(config);
	textList.push_back(info.time);

	//外径
	QString outsideDiameterText = QString("外径: %1 mm").arg(info.outsideDiameter, 0, 'f', 2);
	textList.push_back(outsideDiameterText);

	//孔数
	QString holeCountText = QString("孔数: %1").arg(info.holeCount);
	textList.push_back(holeCountText);

	QString holeDiameterText = QString("孔径:");

	//孔径
	for (const auto& item : info.aperture)
	{
		holeDiameterText.append(QString::number(item, 'f', 2) + " ");
	}
	holeDiameterText.append(" mm");
	textList.push_back(holeDiameterText);

	//孔心距
	if (info.holeCentreDistance.empty())
	{
		textList.push_back("孔心距: 非标纽扣 ");
	}
	else
	{
		QString holeCentreDistanceText = QString("孔心距: %1").arg(info.holeCentreDistance[0], 0, 'f', 2);
		for (size_t i = 1; i < info.holeCentreDistance.size(); i++)
		{
			holeCentreDistanceText.append(QString(" %1").arg(info.holeCentreDistance[i], 0, 'f', 2));
		}
		holeCentreDistanceText.append(" mm");
		textList.push_back(holeCentreDistanceText);
	}

	//rgb
	QString rgbText = QString("R: %1").arg(info.special_R, 0, 'f', 2);
	rgbText.append(QString(" G: %1").arg(info.special_G, 0, 'f', 2));
	rgbText.append(QString(" B: %1").arg(info.special_B, 0, 'f', 2));
	textList.push_back(rgbText);

	//larget rgb
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;

	QString largeColorText = QString("large R: %1 G: %2 B: %3").arg(info.large_R, 0, 'f', 2).arg(info.large_G, 0, 'f', 2).arg(info.large_B, 0, 'f', 2);
	textList.push_back(largeColorText);


	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList);
}

void ImageProcessor::drawButtonDefectInfoText_defect(QImage& image, const ButtonDefectInfo& info)
{
	QVector<QString> textList;
	std::vector<rw::rqw::ImagePainter::PainterConfig> configList;
	rw::rqw::ImagePainter::PainterConfig config;
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
	configList.push_back(config);
	config.textColor = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Red);
	configList.push_back(config);
	//运行时间
	textList.push_back(info.time);

	appendHolesCountDefectInfo(textList, info);
	appendBodyCountDefectInfo(textList, info);
	appendSpecialColorDefectInfo(textList, info);
	appendEdgeDamageDefectInfo(textList, info);
	appendLargeColorDefectInfo(textList, info);
	appendPoreDectInfo(textList, info);
	appendPaintDectInfo(textList, info);
	appendBlockEyeDectInfo(textList, info);
	appendGrindStoneDectInfo(textList, info);
	appendMaterialHeadDectInfo(textList, info);
	appendCrackDectInfo(textList, info);
	appendBrokenEyeDectInfo(textList, info);

	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList);
}

void ImageProcessor::appendHolesCountDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.holesCountEnable)
	{
		QString holeCountText = QString("孔数: %1").arg(info.holeCount) + QString("目标: %1").arg(static_cast<int>(productSet.holesCountValue));
		textList.push_back(holeCountText);
	}

	if (_isbad && productSet.apertureEnable)
	{
		QString apertureText("孔径: ");
		for (const auto& item : info.aperture)
		{
			apertureText.append(QString("%1 ").arg(item, 0, 'f', 2));
		}
		apertureText.append(QString("mm 目标: %1 mm").arg(productSet.apertureValue + productSet.apertureSimilarity));
		textList.push_back(apertureText);
	}

	if (_isbad && productSet.holeCenterDistanceEnable)
	{
		QString holeCenterDistanceText("孔心距: ");
		for (const auto& item : info.holeCentreDistance)
		{
			holeCenterDistanceText.append(QString("%1 ").arg(item, 0, 'f', 2));
		}
		holeCenterDistanceText.append(QString("mm 目标: %1 mm").arg(productSet.holeCenterDistanceValue + productSet.holeCenterDistanceSimilarity));
		textList.push_back(holeCenterDistanceText);
	}
}

void ImageProcessor::appendBodyCountDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.outsideDiameterEnable)
	{
		QString holeCountText = QString("外径: %1 mm ").arg(info.outsideDiameter, 0, 'f', 2) +
			QString(" 目标: %1 mm").arg(productSet.outsideDiameterValue + productSet.outsideDiameterDeviation, 0, 'f', 2);
		textList.push_back(holeCountText);
	}
}

void ImageProcessor::appendSpecialColorDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.specifyColorDifferenceEnable)
	{
		QString specialColorText = QString("R: %1 G: %2 B: %3").arg(info.special_R, 0, 'f', 2).arg(info.special_G, 0, 'f', 2).arg(info.special_B, 0, 'f', 2);
		textList.push_back(specialColorText);
		specialColorText = QString("目标: R: %1 G: %2 B: %3").arg(productSet.specifyColorDifferenceR).arg(productSet.specifyColorDifferenceG).arg(productSet.specifyColorDifferenceB);
		textList.push_back(specialColorText);
	}
}

void ImageProcessor::appendLargeColorDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.largeColorDifferenceEnable)
	{
		QString largeColorText = QString("R: %1 G: %2 B: %3").arg(info.special_R, 0, 'f', 2).arg(info.special_G, 0, 'f', 2).arg(info.special_B, 0, 'f', 2);
		textList.push_back(largeColorText);
		largeColorText = QString("目标: R: %1 G: %2 B: %3").arg(info.large_R).arg(info.large_G).arg(info.large_B);
		textList.push_back(largeColorText);
	}
}

void ImageProcessor::appendEdgeDamageDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.edgeDamageEnable)
	{
		QString edgeDamageText("破边:");
		for (const auto& item : info.edgeDamage)
		{
			edgeDamageText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		edgeDamageText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.edgeDamageSimilarity)));
		textList.push_back(edgeDamageText);
	}
}

void ImageProcessor::appendPoreDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.poreEnable)
	{
		QString poreText("气孔:");
		for (const auto& item : info.pore)
		{
			poreText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		poreText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.poreEnableScore)));
		textList.push_back(poreText);
	}
}

void ImageProcessor::appendPaintDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.paintEnable)
	{
		QString paintText("油漆:");
		for (const auto& item : info.paint)
		{
			paintText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		paintText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.paintEnableScore)));
		textList.push_back(paintText);
	}
}

void ImageProcessor::appendBrokenEyeDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.brokenEyeEnable)
	{
		QString brokenEyeText("破眼:");
		for (const auto& item : info.brokenEye)
		{
			brokenEyeText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		brokenEyeText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.brokenEyeSimilarity)));
		textList.push_back(brokenEyeText);
	}
}

void ImageProcessor::appendCrackDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.crackEnable)
	{
		QString crackText("裂痕:");
		for (const auto& item : info.crack)
		{
			crackText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		crackText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.crackSimilarity)));
		textList.push_back(crackText);
	}
}

void ImageProcessor::appendGrindStoneDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.grindStoneEnable)
	{
		QString grindStoneText("磨石:");
		for (const auto& item : info.grindStone)
		{
			grindStoneText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		grindStoneText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.grindStoneEnableScore)));
		textList.push_back(grindStoneText);
	}
}

void ImageProcessor::appendBlockEyeDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.blockEyeEnable)
	{
		QString blockEyeText("堵眼:");
		for (const auto& item : info.blockEye)
		{
			blockEyeText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		blockEyeText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.blockEyeEnableScore)));
		textList.push_back(blockEyeText);
	}
}

void ImageProcessor::appendMaterialHeadDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.materialHeadEnable)
	{
		QString materialHeadText("料头:");
		for (const auto& item : info.materialHead)
		{
			materialHeadText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		materialHeadText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.materialHeadEnableScore)));
		textList.push_back(materialHeadText);
	}
}

void ImageProcessor::drawShieldingRange(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() != 1)
	{
		return;
	}

	double currentPixelEquivalent = 0;
	if (imageProcessingModuleIndex == 1)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent2;
	}
	else if (imageProcessingModuleIndex == 3)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent3;
	}
	else if (imageProcessingModuleIndex == 4)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent4;
	}

	auto& globalStruct = GlobalStructData::getInstance();
	auto& productSet = globalStruct.dlgProductSetConfig;
	int outerRadius = productSet.outerRadius / 2 / currentPixelEquivalent;
	int innerRadius = productSet.innerRadius / 2 / currentPixelEquivalent;

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Circle;

	auto& body = processResult[processIndex[0]];
	auto out = body.width / 2 * currentPixelEquivalent;
	QPoint centralPoint{ body.center_x,body.center_y };

	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Blue);
	rw::rqw::ImagePainter::drawShapesOnSourceImg(image, centralPoint, outerRadius, config);

	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Magenta);
	rw::rqw::ImagePainter::drawShapesOnSourceImg(image, centralPoint, innerRadius, config);
}

void ImageProcessor::drawErrorRec(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex)
{
	if (processResult.size() == 0)
	{
		return;
	}
	if (processIndex[ClassId::Body].size() == 0)
	{
		return;
	}

	rw::rqw::ImagePainter::PainterConfig config;
	config.thickness = 3;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);
	for (size_t i = 0; i < processIndex.size(); i++)
	{
		if (i == ClassId::Body || i == ClassId::Hole)
		{
			continue;
		}
		for (size_t j = 0; j < processIndex[i].size(); j++)
		{
			auto& item = processResult[processIndex[i][j]];

			switch (item.classId)
			{
			case ClassId::baibian:
				config.text = "白边 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::duyan:
				config.text = "堵眼 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::pobian:
				config.text = "破边 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::qikong:
				config.text = "气孔 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::moshi:
				config.text = "磨石 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::liaotou:
				config.text = "料头 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::zangwu:
				config.text = "赃物 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::pokong:
				config.text = "破孔 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::poyan:
				config.text = "破眼 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::xiaoqikong:
				config.text = "小气孔 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::mofa:
				config.text = "毛发 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::xiaopobian:
				config.text = "小破边 " + QString::number(qRound(item.score * 100));
				break;
			default:
				config.text = QString::number(item.classId) + QString::number(qRound(item.score * 100));
				break;
			}

			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
		}
	}
}

void ImageProcessor::drawErrorRec_error(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	auto& mainWindowConfig = GlobalStructData::getInstance().mainWindowConfig;
	if (processResult.size() == 0)
	{
		return;
	}
	if (processIndex[ClassId::Body].size() == 0)
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
		if (i == ClassId::Body || i == ClassId::Hole)
		{
			continue;
		}

		for (size_t j = 0; j < processIndex[i].size(); j++)
		{
			auto& item = processResult[processIndex[i][j]];
			switch (item.classId)
			{
			case ClassId::baibian:
				config.text = "白边 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::duyan:
				config.text = "堵眼 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::pobian:
				config.text = "破边 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::qikong:
				config.text = "气孔 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::moshi:
				config.text = "磨石 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::liaotou:
				config.text = "料头 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::zangwu:
				config.text = "赃物 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::pokong:
				config.text = "破孔 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::poyan:
				config.text = "破眼 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::xiaoqikong:
				config.text = "小气孔 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::mofa:
				config.text = "毛发 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::xiaopobian:
				config.text = "小破边 " + QString::number(qRound(item.score * 100));
				break;
			default:
				config.text = QString::number(item.classId) + QString::number(qRound(item.score * 100));
				break;
			}

			if (mainWindowConfig.isDefect)
			{
				if (i == ClassId::pobian && productSet.edgeDamageEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::qikong && productSet.poreEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::duyan && productSet.blockEyeEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::moshi && productSet.grindStoneEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::liaotou && productSet.materialHeadEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::zangwu && productSet.paintEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::pokong && productSet.crackEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
				else if (i == ClassId::poyan && productSet.brokenEyeEnable)
				{
					rw::rqw::ImagePainter::drawShapesOnSourceImg(image, item, config);
				}
			}


		}
	}
}

std::vector<std::vector<size_t>>
ImageProcessor::getIndexInBoundary
(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index)
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

bool ImageProcessor::isInBoundary(const rw::DetectionRectangleInfo& info)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto x = info.center_x;
	if (imageProcessingModuleIndex == 1)
	{
		auto& lineLeft = globalStruct.dlgProduceLineSetConfig.limit1;
		auto lineRight = lineLeft + (globalStruct.dlgProductSetConfig.outsideDiameterValue / globalStruct.dlgProduceLineSetConfig.pixelEquivalent1);
		if (lineLeft < x && x < lineRight)
		{
			return true;
		}
	}
	else if (imageProcessingModuleIndex == 2)
	{
		auto& lineRight = globalStruct.dlgProduceLineSetConfig.limit2;
		auto lineLeft = lineRight - (globalStruct.dlgProductSetConfig.outsideDiameterValue / globalStruct.dlgProduceLineSetConfig.pixelEquivalent2);
		if (lineLeft < x && x < lineRight)
		{
			return true;
		}
	}
	else if (imageProcessingModuleIndex == 3)
	{
		auto& lineLeft = globalStruct.dlgProduceLineSetConfig.limit3;
		auto lineRight = lineLeft + (globalStruct.dlgProductSetConfig.outsideDiameterValue / globalStruct.dlgProduceLineSetConfig.pixelEquivalent3);
		if (lineLeft < x && x < lineRight)
		{
			return true;
		}
	}
	else if (imageProcessingModuleIndex == 4)
	{
		auto& lineRight = globalStruct.dlgProduceLineSetConfig.limit4;
		auto lineLeft = lineRight - (globalStruct.dlgProductSetConfig.outsideDiameterValue / globalStruct.dlgProduceLineSetConfig.pixelEquivalent4);
		if (lineLeft < x && x < lineRight)
		{
			return true;
		}
	}
	return false;
}

std::vector<std::vector<size_t>> ImageProcessor::getIndexInShieldingRange(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index) const
{
	std::vector<std::vector<size_t>> result;
	result.resize(index.size());
	if (info.size() == 0)
	{
		return result;
	}

	if (index[ClassId::Body].size() != 1)
	{
		return result;
	}

	double currentPixelEquivalent = 0;
	if (imageProcessingModuleIndex == 1)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent2;
	}
	else if (imageProcessingModuleIndex == 3)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent3;
	}
	else if (imageProcessingModuleIndex == 4)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent4;
	}

	auto& globalStruct = GlobalStructData::getInstance();
	auto& productSet = globalStruct.dlgProductSetConfig;
	int outerRadius = productSet.outerRadius / 2 / currentPixelEquivalent;
	int innerRadius = productSet.innerRadius / 2 / currentPixelEquivalent;

	auto& body = info[index[ClassId::Body][0]];
	QPoint centralPoint{ body.center_x,body.center_y };

	for (int i = 0;i < index.size();i++)
	{
		if (i == ClassId::Body)
		{
			result[i].emplace_back(index[i][0]);
			continue;
		}
		for (int j = 0;j < index[i].size();j++)
		{
			auto& item = info[index[i][j]];
			QPoint targetPoint{ item.center_x,item.center_y };
			if (!isInShieldRange(centralPoint, outerRadius, centralPoint, innerRadius, targetPoint))
			{
				result[i].emplace_back(index[i][j]);
			}
		}
	}

	return result;
}

bool ImageProcessor::isInShieldRange(const QPoint& outCentral, int outR, const QPoint& innerCentral, int innerR,
	const QPoint& point)
{
	auto outDistance = std::sqrt(std::pow(point.x() - outCentral.x(), 2) + std::pow(point.y() - outCentral.y(), 2));
	auto innerDistance = std::sqrt(std::pow(point.x() - innerCentral.x(), 2) + std::pow(point.y() - innerCentral.y(), 2));
	if (outDistance <= outR && innerDistance >= innerR)
	{
		return true;
	}
	return false;
}

void ImageProcessor::clearLargeRGBList()
{
	large_G_list.clear();
	large_R_list.clear();
	large_B_list.clear();
}

ImageProcessor::ImageProcessor(QQueue<MatInfo>& queue, QMutex& mutex, QWaitCondition& condition, int workIndex, QObject* parent)
	: QThread(parent), _queue(queue), _mutex(mutex), _condition(condition), _workIndex(workIndex) {
}

void ImageProcessor::run()
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


		auto& globalData = GlobalStructData::getInstance();

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

		//QVector<QString> processInfo;
		//processInfo.reserve(20);

		//std::vector<rw::imeot::ProcessRectanglesResultOT> vecRecogResultBad;
		//std::vector<rw::imeot::ProcessRectanglesResultOT> vecRecogResultTarget;

		//// 开始计时
		//auto startTime = std::chrono::high_resolution_clock::now();

		////预留处理时间的位子
		//processInfo.emplace_back();

		//// 调用 processAI 函数
		//cv::Mat result = processAI(frame, processInfo, vecRecogResultBad, vecRecogResultTarget);

		//// 结束计时
		//auto endTime = std::chrono::high_resolution_clock::now();

		//// 计算耗时
		//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		//processInfo[0] = QString("处理时间: %1 ms").arg(duration);

		//auto  image = cvMatToQImage(result);

		//// 绘制错误信息
		//ImagePainter::drawTextOnImage(image, processInfo, { ImagePainter::Color::Green,ImagePainter::Color::Red }, 0.07);

		//// 绘制错误定位全局错误定位
		//drawErrorLocate(image, vecRecogResultBad, Qt::green);

		//// 绘制错误定位目标定位
		//drawErrorLocate(image, vecRecogResultTarget, Qt::red);

		//drawLine(image);

		//if (GlobalStructData::getInstance().isTakePictures && _isbad) {
		//	GlobalStructData::getInstance().imageSaveEngine->pushImage(image, "Mark", "Button");
		//}
	}
}

std::vector<std::vector<size_t>> ImageProcessUtilty::getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info)
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

void ImageProcessUtilty::drawHole(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index)
{
	rw::rqw::ImagePainter::PainterConfig config;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Circle;
	config.thickness = 5;

	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::LightBrown);
	for (const auto& item : index)
	{
		rw::rqw::ImagePainter::drawShapesOnSourceImg(image, processResult[item], config);
	}
}

void ImageProcessUtilty::drawBody(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index)
{
	rw::rqw::ImagePainter::PainterConfig config;
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Circle;
	config.thickness = 5;

	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Gray);
	for (const auto& item : index)
	{
		rw::rqw::ImagePainter::drawShapesOnSourceImg(image, processResult[item], config);
	}
}

std::vector<std::vector<size_t>> ImageProcessUtilty::getAllIndexInMaxBody(const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, size_t deviationValue)
{
	std::vector<std::vector<size_t>> result;
	result.resize(index.size());

	auto maxIndex = rw::DetectionRectangleInfo::getMaxAreaRectangleIterator(processResult, index[ClassId::Body]);
	if (maxIndex == processResult.end())
	{
		return result;
	}

	auto bodyIndex = std::distance(processResult.begin(), maxIndex);
	result[ClassId::Body].emplace_back(bodyIndex);

	auto& bodyRec = processResult[bodyIndex];
	for (int i = 0;i < index.size();i++)
	{
		if (i == ClassId::Body)
		{
			continue;
		}
		for (int j = 0;j < index[i].size();j++)
		{
			auto& currentRec = processResult[index[i][j]];
			if (bodyRec.leftTop.first - deviationValue <= currentRec.center_x && currentRec.center_x <= bodyRec.rightBottom.first + deviationValue)
			{
				if (bodyRec.leftTop.second - deviationValue <= currentRec.center_y && bodyRec.rightBottom.second + deviationValue >= currentRec.center_y)
				{
					result[i].emplace_back(index[i][j]);
				}
			}
		}
	}
	return result;
}


cv::Vec3f ImageProcessUtilty::calculateRegionRGB(const cv::Mat& image, const rw::DetectionRectangleInfo& total,
	CropMode mode, const std::vector<size_t>& index, const std::vector<rw::DetectionRectangleInfo>& processResult,
	CropMode excludeMode)
{
	if (image.empty()) {
		return cv::Vec3f(0, 0, 0);
	}
	if (image.channels() != 3) {
		return cv::Vec3f(0, 0, 0);
	}
	cv::Rect rect_total(
		cv::Point(total.leftTop.first, total.leftTop.second),
		cv::Size(total.width, total.height)
	);
	std::vector<cv::Rect> rect_exclude;
	for (const auto& item : index) {
		cv::Rect rect_excludeItem(
			cv::Point(processResult[item].leftTop.first, processResult[item].leftTop.second),
			cv::Size(processResult[item].width, processResult[item].height)
		);
		rect_exclude.push_back(rect_excludeItem);
	}
	return calculateRegionRGB(image, rect_total, mode, rect_exclude, excludeMode);
}


void ImageProcessor::run_debug(MatInfo& frame)
{
	//AI开始识别
	ButtonDefectInfo defectInfo;
	auto startTime = std::chrono::high_resolution_clock::now();

	auto processResult = _modelEngineOT->processImg(frame.image);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	defectInfo.time = QString("处理时间: %1 ms").arg(duration);
	//AI识别完成
	//过滤出有效索引
	auto processResultIndex = filterEffectiveIndexes_debug(processResult);
	//获取到当前图像的缺陷信息
	getEliminationInfo_debug(defectInfo, processResult, processResultIndex, frame.image);

	//绘制defect信息
	auto  image = cvMatToQImage(frame.image);
	drawButtonDefectInfoText(image, defectInfo);

	//绘制识别框
	drawVerticalBoundaryLine(image);
	drawShieldingRange(image, processResult, processResultIndex[ClassId::Body]);
	drawErrorRec(image, processResult, processResultIndex);
	ImageProcessUtilty::drawHole(image, processResult, processResultIndex[ClassId::Hole]);
	ImageProcessUtilty::drawBody(image, processResult, processResultIndex[ClassId::Body]);

	QPixmap pixmap = QPixmap::fromImage(image);
	emit imageReady(pixmap);
}

void ImageProcessor::run_monitor(MatInfo& frame)
{
	auto  image = cvMatToQImage(frame.image);
	QPixmap pixmap = QPixmap::fromImage(image);
	emit imageReady(pixmap);
}

void ImageProcessor::run_OpenRemoveFunc(MatInfo& frame)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;

	//AI开始识别
	ButtonDefectInfo defectInfo;
	auto startTime = std::chrono::high_resolution_clock::now();

	auto processResult = _modelEngineOT->processImg(frame.image);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	defectInfo.time = QString("处理时间: %1 ms").arg(duration);
	//AI识别完成
	//过滤出有效索引
	auto processResultIndex = filterEffectiveIndexes_defect(processResult);

	//获取到当前图像的缺陷信息
	getEliminationInfo_defect(defectInfo, processResult, processResultIndex, frame.image);

	//剔除逻辑获取_isbad
	run_OpenRemoveFunc_process_defect_info(defectInfo);

	//如果_isbad为true，将错误信息发送到剔除队列中
	run_OpenRemoveFunc_emitErrorInfo(frame);

	//绘制defect信息
	auto  image = cvMatToQImage(frame.image);
	drawButtonDefectInfoText_defect(image, defectInfo);

	//绘制识别框
	drawVerticalBoundaryLine(image);
	if (productSet.shieldingRangeEnable)
	{
		drawShieldingRange(image, processResult, processResultIndex[ClassId::Body]);
	}
	drawErrorRec(image, processResult, processResultIndex);
	drawErrorRec_error(image, processResult, processResultIndex);
	ImageProcessUtilty::drawHole(image, processResult, processResultIndex[ClassId::Hole]);
	ImageProcessUtilty::drawBody(image, processResult, processResultIndex[ClassId::Body]);

	//保存图像
	if (globalData.isTakePictures) {
		if (_isbad) {
			globalData.imageSaveEngine->pushImage(cvMatToQImage(frame.image), "NG", "Button");
			globalData.imageSaveEngine->pushImage(image, "Mask", "Button");
		}
		else {
			globalData.imageSaveEngine->pushImage(cvMatToQImage(frame.image), "OK", "Button");
		}
	}

	QPixmap pixmap = QPixmap::fromImage(image);
	emit imageReady(pixmap);
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info(const ButtonDefectInfo& info)
{
	_isbad = false;
	auto& globalData = GlobalStructData::getInstance();
	auto& isOpenDefect = globalData.mainWindowConfig.isDefect;
	if (isOpenDefect)
	{
		run_OpenRemoveFunc_process_defect_info_hole(info);
		run_OpenRemoveFunc_process_defect_info_body(info);
		run_OpenRemoveFunc_process_defect_info_specialColor(info);
		run_OpenRemoveFunc_process_defect_info_edgeDamage(info);
		run_OpenRemoveFunc_process_defect_info_pore(info);
		run_OpenRemoveFunc_process_defect_info_paint(info);
		run_OpenRemoveFunc_process_defect_info_brokenEye(info);
		run_OpenRemoveFunc_process_defect_info_blockEye(info);
		run_OpenRemoveFunc_process_defect_info_grindStone(info);
		run_OpenRemoveFunc_process_defect_info_materialHead(info);
		run_OpenRemoveFunc_process_defect_info_largeColor(info);
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_hole(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	//孔数
	if (productSet.holesCountEnable)
	{
		auto& holeCount = info.holeCount;
		if (holeCount != static_cast<size_t>(productSet.holesCountValue))
		{
			_isbad = true;
		}
	}

	//孔径
	if (productSet.apertureEnable)
	{
		auto& aperture = info.aperture;
		auto apertureStandard = productSet.apertureValue + productSet.apertureSimilarity;
		for (const auto& item : aperture)
		{
			if (item > apertureStandard)
			{
				_isbad = true;
			}
		}
	}

	//孔心距
	if (productSet.holeCenterDistanceEnable)
	{
		auto& holeCentreDistance = info.holeCentreDistance;
		auto holeCentreDistanceStandard = productSet.holeCenterDistanceValue + productSet.holeCenterDistanceSimilarity;
		for (const auto& item : holeCentreDistance)
		{
			if (item > holeCentreDistanceStandard)
			{
				_isbad = true;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_body(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.outsideDiameterEnable)
	{
		auto& outsideDiameterDeviation = productSet.outsideDiameterDeviation;
		auto outsideDiameterStandard = outsideDiameterDeviation + productSet.outsideDiameterValue;
		auto outsideDiameter = info.outsideDiameter;

		if (outsideDiameter > outsideDiameterStandard)
		{
			_isbad = true;
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_specialColor(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.specifyColorDifferenceEnable)
	{
		auto& special_R = info.special_R;
		auto& special_G = info.special_G;
		auto& special_B = info.special_B;
		auto& specialColorDeviation = productSet.specifyColorDifferenceDeviation;
		auto special_R_standard = productSet.specifyColorDifferenceR + specialColorDeviation;
		auto special_G_standard = productSet.specifyColorDifferenceG + specialColorDeviation;
		auto special_B_standard = productSet.specifyColorDifferenceB + specialColorDeviation;
		if (special_R > special_R_standard || special_G > special_G_standard || special_B > special_B_standard)
		{
			_isbad = true;
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_edgeDamage(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.edgeDamageEnable)
	{
		auto& edgeDamage = info.edgeDamage;
		if (edgeDamage.empty())
		{
			return;
		}

		for (const auto& item : edgeDamage)
		{
			if (item > productSet.edgeDamageSimilarity)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const
{
	auto& globalStruct = GlobalStructData::getInstance();

	if (_isbad) {
		++globalStruct.statisticalInfo.wasteCount;
	}

	if (imageProcessingModuleIndex == 2 || imageProcessingModuleIndex == 4) {
		++globalStruct.statisticalInfo.produceCount;
	}

	if (_isbad) {
		float absLocation = frame.location;
		if (absLocation < 0) {
			absLocation = -absLocation;
		}

		switch (imageProcessingModuleIndex)
		{
		case 1:
			globalStruct.productPriorityQueue1.push(absLocation);
			break;
		case 2:
			globalStruct.productPriorityQueue2.push(absLocation);
			break;
		case 3:
			globalStruct.productPriorityQueue3.push(absLocation);
			break;
		case 4:
			globalStruct.productPriorityQueue4.push(absLocation);
			break;
		default:
			break;
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_pore(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.poreEnable)
	{
		auto& pore = info.pore;
		if (pore.empty())
		{
			return;
		}
		for (const auto& item : pore)
		{
			if (item > productSet.poreEnableScore)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_paint(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.paintEnable)
	{
		auto& paint = info.paint;
		if (paint.empty())
		{
			return;
		}
		for (const auto& item : paint)
		{
			if (item > productSet.paintEnableScore)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_brokenEye(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.brokenEyeEnable)
	{
		auto& brokenEye = info.brokenEye;
		if (brokenEye.empty())
		{
			return;
		}
		for (const auto& item : brokenEye)
		{
			if (item > productSet.brokenEyeSimilarity)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_crack(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.crackEnable)
	{
		auto& crack = info.crack;
		if (crack.empty())
		{
			return;
		}
		for (const auto& item : crack)
		{
			if (item > productSet.crackSimilarity)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_grindStone(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.grindStoneEnable)
	{
		auto& grindStone = info.grindStone;
		if (grindStone.empty())
		{
			return;
		}
		for (const auto& item : grindStone)
		{
			if (item > productSet.grindStoneEnableScore)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_blockEye(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.blockEyeEnable)
	{
		auto& blockEye = info.blockEye;
		if (blockEye.empty())
		{
			return;
		}
		for (const auto& item : blockEye)
		{
			if (item > productSet.blockEyeEnableScore)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_materialHead(const ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.materialHeadEnable)
	{
		auto& materialHead = info.materialHead;
		if (materialHead.empty())
		{
			return;
		}
		for (const auto& item : materialHead)
		{
			if (item > productSet.materialHeadEnableScore)
			{
				_isbad = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_largeColor(const ButtonDefectInfo& info)
{

	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (large_G_list.empty() || large_R_list.empty() || large_B_list.empty())
	{
		return;
	}
	if (productSet.largeColorDifferenceEnable)
	{
		auto& deviation = productSet.largeColorDifferenceDeviation;

		auto largeR = info.large_R + deviation;
		auto largeG = info.large_G + deviation;
		auto largeB = info.large_B + deviation;


		if (info.special_R > largeR || info.special_G > largeG || info.special_B > largeB)
		{
			_isbad = true;
		}
	}
}

void ImageProcessor::getEliminationInfo_debug(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const
	cv::Mat& mat)
{
	getHoleInfo(info, processResult, index[ClassId::Hole]);
	getBodyInfo(info, processResult, index[ClassId::Body]);
	getEdgeDamageInfo(info, processResult, index[ClassId::pobian]);
	getPoreInfo(info, processResult, index[ClassId::qikong]);
	getBlockEyeInfo(info, processResult, index[ClassId::duyan]);
	getGrindStoneInfo(info, processResult, index[ClassId::moshi]);
	getMaterialHeadInfo(info, processResult, index[ClassId::liaotou]);
	getPaintInfo(info, processResult, index[ClassId::zangwu]);
	getCrackInfo(info, processResult, index[ClassId::pokong]);
	getBrokenEyeInfo(info, processResult, index[ClassId::poyan]);
	getPaintInfo(info, processResult, index[ClassId::mofa]);
	getLargeColorDifference(info, processResult, index, mat);
	getSpecialColorDifference(info, processResult, index, mat);
}

void ImageProcessor::getEliminationInfo_defect(ButtonDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index,
	const cv::Mat& mat)
{

	getHoleInfo(info, processResult, index[ClassId::Hole]);
	getBodyInfo(info, processResult, index[ClassId::Body]);
	getEdgeDamageInfo(info, processResult, index[ClassId::pobian]);
	getPoreInfo(info, processResult, index[ClassId::qikong]);
	getBlockEyeInfo(info, processResult, index[ClassId::duyan]);
	getGrindStoneInfo(info, processResult, index[ClassId::moshi]);
	getMaterialHeadInfo(info, processResult, index[ClassId::liaotou]);
	getPaintInfo(info, processResult, index[ClassId::zangwu]);
	getCrackInfo(info, processResult, index[ClassId::pokong]);
	getBrokenEyeInfo(info, processResult, index[ClassId::poyan]);
	getPaintInfo(info, processResult, index[ClassId::mofa]);
	getLargeColorDifference(info, processResult, index, mat);
	getSpecialColorDifference(info, processResult, index, mat);
}

void ImageProcessor::getHoleInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	info.holeCount = processIndex.size();
	double currentPixelEquivalent = 0;
	if (imageProcessingModuleIndex == 1)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent2;
	}
	else if (imageProcessingModuleIndex == 3)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent3;
	}
	else if (imageProcessingModuleIndex == 4)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent4;
	}

	//计算孔径
	for (const auto& item : processIndex)
	{
		double aperture = processResult[item].width * currentPixelEquivalent;
		info.aperture.emplace_back(aperture);
	}

	//计算孔心距 其他孔非标处理
	if (processIndex.size() % 2 != 0)
	{
		//奇数孔忽略
		return;
	}

	//两孔
	if (processIndex.size() == 2)
	{
		auto x = std::abs(processResult[0].center_x - processResult[1].center_x);
		auto y = std::abs(processResult[0].center_y - processResult[1].center_y);
		auto holeCentreDistance = std::sqrt(std::pow(x, 2) + std::pow(y, 2)) * currentPixelEquivalent;
		info.holeCentreDistance.push_back(holeCentreDistance);
	}

	//四孔
	if (processIndex.size() == 4)
	{
		// 存储 x 和 y 轴的最大差值及对应的孔对
		int maxXDiff = 0, maxYDiff = 0;
		std::pair<size_t, size_t> maxXPair, maxYPair;

		// 遍历所有可能的孔对
		for (size_t i = 0; i < processIndex.size(); ++i)
		{
			for (size_t j = i + 1; j < processIndex.size(); ++j)
			{
				// 获取两个孔的中心坐标
				const auto& hole1 = processResult[processIndex[i]];
				const auto& hole2 = processResult[processIndex[j]];

				int xDiff = std::abs(hole1.center_x - hole2.center_x);
				int yDiff = std::abs(hole1.center_y - hole2.center_y);

				// 更新 x 轴最大差值
				if (xDiff > maxXDiff)
				{
					maxXDiff = xDiff;
					maxXPair = { i, j };
				}

				// 更新 y 轴最大差值
				if (yDiff > maxYDiff)
				{
					maxYDiff = yDiff;
					maxYPair = { i, j };
				}
			}
		}

		// 计算 x 轴最大差值对应的孔心距
		const auto& holeX1 = processResult[processIndex[maxXPair.first]];
		const auto& holeX2 = processResult[processIndex[maxXPair.second]];
		double xAxisDistance = std::sqrt(
			std::pow(holeX1.center_x - holeX2.center_x, 2) +
			std::pow(holeX1.center_y - holeX2.center_y, 2)) * currentPixelEquivalent;

		// 计算 y 轴最大差值对应的孔心距
		const auto& holeY1 = processResult[processIndex[maxYPair.first]];
		const auto& holeY2 = processResult[processIndex[maxYPair.second]];
		double yAxisDistance = std::sqrt(
			std::pow(holeY1.center_x - holeY2.center_x, 2) +
			std::pow(holeY1.center_y - holeY2.center_y, 2)) * currentPixelEquivalent;

		// 将结果存储到 info.holeCentreDistance
		info.holeCentreDistance.push_back(xAxisDistance);
		info.holeCentreDistance.push_back(yAxisDistance);
	}


}

void ImageProcessor::getBodyInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	double currentPixelEquivalent = 0;
	if (imageProcessingModuleIndex == 1)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent1;
	}
	else if (imageProcessingModuleIndex == 2)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent2;
	}
	else if (imageProcessingModuleIndex == 3)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent3;
	}
	else if (imageProcessingModuleIndex == 4)
	{
		currentPixelEquivalent = GlobalStructData::getInstance().dlgProduceLineSetConfig.pixelEquivalent4;
	}

	for (const auto& item : processIndex)
	{
		double outsideDiameter = processResult[item].width * currentPixelEquivalent;
		info.outsideDiameter = outsideDiameter;
	}
}

void ImageProcessor::getSpecialColorDifference(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat)
{
	if (index[ClassId::Body].empty())
	{
		return;
	}
	auto rgb = ImageProcessUtilty::calculateRegionRGB(mat,
		processResult[index[ClassId::Body][0]],
		ImageProcessUtilty::CropMode::InscribedCircle,
		index[ClassId::Hole],
		processResult,
		ImageProcessUtilty::CropMode::InscribedCircle);
	info.special_R = rgb[0];
	info.special_G = rgb[1];
	info.special_B = rgb[2];
}


void ImageProcessor::getLargeColorDifference(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat)
{
	if (index[ClassId::Body].empty())
	{
		return;
	}
	auto rgb = ImageProcessUtilty::calculateRegionRGB(mat,
		processResult[index[ClassId::Body][0]],
		ImageProcessUtilty::CropMode::InscribedCircle,
		index[ClassId::Hole],
		processResult,
		ImageProcessUtilty::CropMode::InscribedCircle);
	if (large_G_list.size() != 5)
	{
		large_G_list.push_back(rgb[0]);
		large_R_list.push_back(rgb[1]);
		large_B_list.push_back(rgb[2]);
	}
	else
	{
		float sum = std::accumulate(large_G_list.begin(), large_G_list.end(), 0.0f);
		info.large_R = sum / 5;
		sum = std::accumulate(large_R_list.begin(), large_R_list.end(), 0.0f);
		info.large_G = sum / 5;
		sum = std::accumulate(large_B_list.begin(), large_B_list.end(), 0.0f);
		info.large_B = sum / 5;
	}

}

void ImageProcessor::getEdgeDamageInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  edgeDamage = processResult[item].score * 100;
		info.edgeDamage.emplace_back(edgeDamage);
	}
}

void ImageProcessor::getPoreInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  pore = processResult[item].score * 100;
		info.pore.emplace_back(pore);
	}
}

void ImageProcessor::getPaintInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  paint = processResult[item].score * 100;
		info.paint.emplace_back(paint);
	}
}

void ImageProcessor::getBrokenEyeInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  brokenEye = processResult[item].score * 100;
		info.brokenEye.emplace_back(brokenEye);
	}
}

void ImageProcessor::getCrackInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  crack = processResult[item].score * 100;
		info.crack.emplace_back(crack);
	}
}

void ImageProcessor::getGrindStoneInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  grindStone = processResult[item].score * 100;
		info.grindStone.emplace_back(grindStone);
	}
}

void ImageProcessor::getBlockEyeInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  blockEye = processResult[item].score * 100;
		info.blockEye.emplace_back(blockEye);
	}
}

void ImageProcessor::getMaterialHeadInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex)
{
	if (processIndex.size() == 0)
	{
		return;
	}
	for (const auto& item : processIndex)
	{
		auto  materialHead = processResult[item].score * 100;
		info.materialHead.emplace_back(materialHead);
	}
}

void ImageProcessingModule::BuildModule()
{
	for (int i = 0; i < _numConsumers; ++i) {
		static size_t workIndexCount = 0;
		ImageProcessor* processor = new ImageProcessor(_queue, _mutex, _condition, workIndexCount, this);
		workIndexCount++;
		processor->buildModelEngineOT(modelEngineOTPath);
		processor->imageProcessingModuleIndex = index;
		connect(processor, &ImageProcessor::imageReady, this, &ImageProcessingModule::imageReady, Qt::QueuedConnection);
		_processors.push_back(processor);
		processor->start();
	}
}

void ImageProcessingModule::clearLargeRGBList()
{
	for (auto& item : _processors)
	{
		item->clearLargeRGBList();
	}
}

//void ImageProcessingModule::reloadOOModel()
//{
//	for (auto& item : _processors)
//	{
//		item->buildModelEngineOnnxOO(modelOnnxOOPath, modelNamePath);
//	}
//}
//
//void ImageProcessingModule::reloadSOModel()
//{
//	for (auto& item : _processors)
//	{
//		item->buildModelEngineOnnxSO(modelOnnxSOPath, modelNamePath);
//	}
//}

ImageProcessingModule::ImageProcessingModule(int numConsumers, QObject* parent)
	: QObject(parent), _numConsumers(numConsumers) {
}

ImageProcessingModule::~ImageProcessingModule()
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

void ImageProcessingModule::onFrameCaptured(cv::Mat frame, float location, size_t index)
{
	emit imgForDlgNewProduction(frame, index);

	if (frame.empty()) {
		return; // 跳过空帧
	}

	QMutexLocker locker(&_mutex);
	MatInfo mat;
	mat.image = frame;
	mat.location = location;
	mat.index = index;
	_queue.enqueue(mat);
	_condition.wakeOne();
}

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

cv::Vec3f ImageProcessUtilty::calculateRegionRGB(const cv::Mat& image, const cv::Rect& rect, CropMode mode, std::vector<cv::Rect> excludeRegions, CropMode excludeMode)
{
	// 检查图像是否为空
	if (image.empty()) {
		throw std::invalid_argument("Input image is empty.");
	}

	// 检查图像是否为彩色图像
	if (image.channels() != 3) {
		throw std::invalid_argument("Input image must be a 3-channel (RGB) image.");
	}

	// 检查矩形是否在图像范围内
	cv::Rect imageBounds(0, 0, image.cols, image.rows);
	cv::Rect validRect = rect & imageBounds; // 取交集，确保矩形在图像范围内

	if (validRect.width <= 0 || validRect.height <= 0) {
		throw std::invalid_argument("The rectangle is outside the image bounds.");
	}

	// 创建掩码
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

	// 根据模式标记主区域
	if (mode == CropMode::Rectangle) {
		mask(validRect).setTo(255);
	}
	else if (mode == CropMode::InscribedCircle) {
		int radius = std::min(validRect.width, validRect.height) / 2;
		cv::Point center(validRect.x + validRect.width / 2, validRect.y + validRect.height / 2);
		cv::circle(mask, center, radius, cv::Scalar(255), -1);
	}
	else {
		throw std::invalid_argument("Invalid crop mode.");
	}

	// 处理需要排除的区域
	for (const auto& excludeRect : excludeRegions) {
		// 确保排除区域在主区域内，且不等于主区域
		if ((excludeRect & validRect) != excludeRect || excludeRect == validRect) {
			continue; // 跳过无效的排除区域
		}

		if (excludeMode == CropMode::Rectangle) {
			// 在掩码中去掉矩形区域
			mask(excludeRect).setTo(0);
		}
		else if (excludeMode == CropMode::InscribedCircle) {
			// 计算内接圆的半径和中心点
			int radius = std::min(excludeRect.width, excludeRect.height) / 2;
			cv::Point center(excludeRect.x + excludeRect.width / 2, excludeRect.y + excludeRect.height / 2);
			cv::circle(mask, center, radius, cv::Scalar(0), -1);
		}
	}

	// 使用掩码计算平均 RGB 值
	cv::Scalar meanRGB = cv::mean(image, mask);

	// 返回平均 RGB 值
	return cv::Vec3f(meanRGB[2], meanRGB[1], meanRGB[0]); // 注意：OpenCV 的通道顺序是 BGR
}

