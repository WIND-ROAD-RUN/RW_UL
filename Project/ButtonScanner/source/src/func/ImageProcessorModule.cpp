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
	config.conf_threshold = 0.1f;
	config.nms_threshold = 0.1f;
	config.need_keep_classids={0,1};
	config.modelPath = enginePath.toStdString();
	_modelEngineOT = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_seg, rw::ModelEngineDeployType::TensorRT);
}

void ImageProcessor::buildOnnxRuntimeOO(const QString& enginePath)
{
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.5f;
	config.nms_threshold = 0.5f;
	config.modelPath = enginePath.toStdString();
	_onnxRuntimeOO = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_det, rw::ModelEngineDeployType::TensorRT);
}

void ImageProcessor::reloadOnnxRuntimeOO(const QString& enginePath)
{
	if (_onnxRuntimeOO)
	{
		_onnxRuntimeOO.reset();
		buildOnnxRuntimeOO(enginePath);
	}
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

std::vector<std::vector<size_t>> ImageProcessor::filterEffectiveIndexes_positive(
	std::vector<rw::DetectionRectangleInfo> info)
{
	auto& globalStruct = GlobalStructData::getInstance();

	auto processResultIndex = ImageProcessUtilty::getClassIndex(info);
	processResultIndex = getIndexInBoundary(info, processResultIndex);
	if (globalStruct.dlgProductSetConfig.shieldingRangeEnable)
	{
		processResultIndex = getIndexInShieldingRange(info, processResultIndex);
	}
	return processResultIndex;
}

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

	auto& productSet = GlobalStructData::getInstance().mainWindowConfig;
	auto& positive = productSet.isPositive;
	auto& defect = productSet.isDefect;

	if (positive)
	{
		appendPositiveDectInfo(textList, info);
	}

	if (defect)
	{
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
	}

	rw::rqw::ImagePainter::drawTextOnImage(image, textList, configList);
}

void ImageProcessor::appendHolesCountDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.holesCountEnable&&info.isDrawholeCount)
	{
		QString holeCountText = QString("孔数: %1").arg(info.holeCount) + QString("目标: %1").arg(static_cast<int>(productSet.holesCountValue));
		textList.push_back(holeCountText);
	}

	if (_isbad && productSet.apertureEnable&&info.isDrawaperture)
	{
		QString apertureText("孔径: ");
		for (const auto& item : info.aperture)
		{
			apertureText.append(QString("%1 ").arg(item, 0, 'f', 2));
		}
		apertureText.append(QString("mm 目标: %1 mm").arg(productSet.apertureValue + productSet.apertureSimilarity));
		textList.push_back(apertureText);
	}

	if (_isbad && productSet.holeCenterDistanceEnable && info.isDraweholeCentreDistance)
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
	if (_isbad && productSet.outsideDiameterEnable&& info.isoutsideDiameter)
	{
		QString holeCountText = QString("外径: %1 mm ").arg(info.outsideDiameter, 0, 'f', 2) +
			QString(" 目标: %1 mm").arg(productSet.outsideDiameterValue + productSet.outsideDiameterDeviation, 0, 'f', 2);
		textList.push_back(holeCountText);
	}
}

void ImageProcessor::appendSpecialColorDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.specifyColorDifferenceEnable && info.isDrawSpecialColor)
	{
		QString specialColorText = QString("R: %1 G: %2 B: %3").arg(info.special_R, 0, 'f', 2).arg(info.special_G, 0, 'f', 2).arg(info.special_B, 0, 'f', 2);
		textList.push_back(specialColorText);
		specialColorText = QString("目标: R: %1 G: %2 B: %3 偏差: %4").arg(productSet.specifyColorDifferenceR).arg(productSet.specifyColorDifferenceG).arg(productSet.specifyColorDifferenceB).arg(productSet.specifyColorDifferenceDeviation);
		textList.push_back(specialColorText);
	}
}

void ImageProcessor::appendLargeColorDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.largeColorDifferenceEnable && info.isDrawlargeColor)
	{
		QString largeColorText = QString("R: %1 G: %2 B: %3").arg(info.special_R, 0, 'f', 2).arg(info.special_G, 0, 'f', 2).arg(info.special_B, 0, 'f', 2);
		textList.push_back(largeColorText);
		largeColorText = QString("目标: R: %1 G: %2 B: %3 偏差: %4").arg(info.large_R).arg(info.large_G).arg(info.large_B).arg(productSet.largeColorDifferenceDeviation);
		textList.push_back(largeColorText);
	}
}

void ImageProcessor::appendEdgeDamageDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.edgeDamageEnable && info.isDrawedgeDamage)
	{
		auto targetScore = static_cast<int>(productSet.edgeDamageSimilarity);
		QString edgeDamageText("破边:");
		for (const auto& item : info.edgeDamage)
		{
			edgeDamageText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		edgeDamageText.append(QString(" 目标: %1").arg(targetScore));
		textList.push_back(edgeDamageText);
	}
}

void ImageProcessor::appendPoreDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.poreEnable && info.isDrawpore)
	{
		QString poreText("气孔:");
		for (const auto& item : info.pore)
		{
			poreText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		poreText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.poreEnableScore)));
		textList.push_back(poreText);
	}
}

void ImageProcessor::appendPaintDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.paintEnable && info.isDrawpaint)
	{
		QString paintText("油漆:");
		for (const auto& item : info.paint)
		{
			paintText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		paintText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.paintEnableScore)));
		textList.push_back(paintText);
	}
}

void ImageProcessor::appendBrokenEyeDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.brokenEyeEnable && info.isDrawbrokenEye)
	{
		QString brokenEyeText("破眼:");
		for (const auto& item : info.brokenEye)
		{
			brokenEyeText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		brokenEyeText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.brokenEyeSimilarity)));
		textList.push_back(brokenEyeText);
	}
}

void ImageProcessor::appendPositiveDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().mainWindowConfig;
	auto & dlgSet= GlobalStructData::getInstance().dlgHideScoreSetConfig;
	if (_isbad && productSet.isPositive && info.isDrawpositiver)
	{
		QString positiveText("正反:");
		for (const auto& item : info.positive)
		{
			positiveText.push_back(QString(" %1 ").arg(static_cast<int>(item)));
		}
		positiveText.append(QString(" 目标: %1").arg(static_cast<int>(dlgSet.forAndAgainstScore)));
		textList.push_back(positiveText);
	}
}

void ImageProcessor::appendCrackDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.crackEnable && info.isDrawcrack)
	{
		QString crackText("裂痕:");
		for (const auto& item : info.crack)
		{
			crackText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		crackText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.crackSimilarity)));
		textList.push_back(crackText);
	}
}

void ImageProcessor::appendGrindStoneDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.grindStoneEnable && info.isDrawgrindStone)
	{
		QString grindStoneText("磨石:");
		for (const auto& item : info.grindStone)
		{
			grindStoneText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		grindStoneText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.grindStoneEnableScore)));
		textList.push_back(grindStoneText);
	}
}

void ImageProcessor::appendBlockEyeDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.blockEyeEnable && info.isDrawblockEye)
	{
		QString blockEyeText("堵眼:");
		for (const auto& item : info.blockEye)
		{
			blockEyeText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
		}
		blockEyeText.append(QString(" 目标: %1").arg(static_cast<int>(productSet.blockEyeEnableScore)));
		textList.push_back(blockEyeText);
	}
}

void ImageProcessor::appendMaterialHeadDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info)
{
	auto& productSet = GlobalStructData::getInstance().dlgProductSetConfig;
	if (_isbad && productSet.materialHeadEnable && info.isDrawmaterialHead)
	{
		QString materialHeadText("料头:");
		for (const auto& item : info.materialHead)
		{
			materialHeadText.push_back(QString(" %1 ").arg(item, 0, 'f', 2));
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
				config.text = "油漆 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::liehen:
				config.text = "裂痕 " + QString::number(qRound(item.score * 100));
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
				config.text = "油漆 " + QString::number(qRound(item.score * 100));
				break;
			case ClassId::liehen:
				config.text = "裂痕 " + QString::number(qRound(item.score * 100));
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
				else if (i == ClassId::liehen && productSet.crackEnable)
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
			if (info[index[i][j]].classId==ClassId::Body)
			{
				if (isInBoundary(info[index[i][j]]))
				{
					result[i].push_back(index[i][j]);
				}
			}
			else
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
			auto leftStandard = static_cast<int>(bodyRec.leftTop.first - static_cast<int>(deviationValue));
			auto rightStandard = static_cast<int>(bodyRec.rightBottom.first + static_cast<int>(deviationValue));
			if (leftStandard <= currentRec.center_x && currentRec.center_x <= rightStandard)
			{
				auto leftStandard = static_cast<int>(bodyRec.leftTop.second - static_cast<int>(deviationValue));
				auto rightStandard = static_cast<int>(bodyRec.rightBottom.second + static_cast<int>(deviationValue));
				if (leftStandard <= currentRec.center_y && rightStandard >= currentRec.center_y)
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
	auto isPositive = globalData.mainWindowConfig.isPositive;
	auto isDefect = globalData.mainWindowConfig.isDefect;

	//AI开始识别
	ButtonDefectInfo defectInfo;
	auto startTime = std::chrono::high_resolution_clock::now();

	QFuture<std::vector<std::vector<size_t>>> futureResultIndex;

	std::vector<rw::DetectionRectangleInfo> processResultPositive;

	if (isPositive)
	{
		// 使用 QtConcurrent::run 将处理逻辑放到单独的线程中
		futureResultIndex = QtConcurrent::run([this, &processResultPositive,&frame]() {
			processResultPositive = _onnxRuntimeOO->processImg(frame.image);
			//过滤出有效索引
			return filterEffectiveIndexes_positive(processResultPositive);
			});
	}

	std::vector<rw::DetectionRectangleInfo> processResultDefect;
	if (isDefect) {
		processResultDefect = _modelEngineOT->processImg(frame.image);
	}

	if (isPositive)
	{
		futureResultIndex.waitForFinished();
		auto processResultIndexOO = futureResultIndex.result();
		getEliminationInfo_positive(defectInfo, processResultPositive, processResultIndexOO, frame.image);
	}


	//过滤出有效索引
	auto processResultIndex = filterEffectiveIndexes_defect(processResultDefect);

	//获取到当前图像的缺陷信息
	getEliminationInfo_defect(defectInfo, processResultDefect, processResultIndex, frame.image);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	defectInfo.time = QString("处理时间: %1 ms").arg(duration);
	//AI识别完成

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
		drawShieldingRange(image, processResultDefect, processResultIndex[ClassId::Body]);
	}
	drawErrorRec(image, processResultDefect, processResultIndex);
	drawErrorRec_error(image, processResultDefect, processResultIndex);
	ImageProcessUtilty::drawHole(image, processResultDefect, processResultIndex[ClassId::Hole]);
	ImageProcessUtilty::drawBody(image, processResultDefect, processResultIndex[ClassId::Body]);

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

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_positive(ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& hideConfig = globalData.dlgHideScoreSetConfig;
	for (const auto & item:info.positive)
	{
		if (item >= hideConfig.forAndAgainstScore)
		{
			_isbad = true;
			info.isDrawpositiver = true;
			break;
		}

	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info(ButtonDefectInfo& info)
{
	_isbad = false;
	auto& globalData = GlobalStructData::getInstance();
	auto& isOpenDefect = globalData.mainWindowConfig.isDefect;
	auto & isOpenPositive= globalData.mainWindowConfig.isPositive;
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
		run_OpenRemoveFunc_process_defect_info_crack(info);
	}

	if (isOpenPositive)
	{
		run_OpenRemoveFunc_process_defect_info_positive(info);
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_hole( ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	//孔数
	if (productSet.holesCountEnable)
	{
		auto& holeCount = info.holeCount;
		if (holeCount != static_cast<size_t>(productSet.holesCountValue))
		{
			info.isDrawholeCount = true;
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
				info.isDrawaperture = true;
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
				info.isDraweholeCentreDistance = true;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_body( ButtonDefectInfo& info)
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
			info.isoutsideDiameter = true;
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_specialColor( ButtonDefectInfo& info)
{
	auto& globalData = GlobalStructData::getInstance();
	auto& productSet = globalData.dlgProductSetConfig;
	if (productSet.specifyColorDifferenceEnable)
	{
		auto& special_R = info.special_R;
		auto& special_G = info.special_G;
		auto& special_B = info.special_B;
		auto& specialColorDeviation = productSet.specifyColorDifferenceDeviation;
		auto special_R_standard = productSet.specifyColorDifferenceR;
		auto special_G_standard = productSet.specifyColorDifferenceG;
		auto special_B_standard = productSet.specifyColorDifferenceB;
		auto isInR = ((special_R_standard - specialColorDeviation) <= special_R) &&( special_R <= (special_R_standard + specialColorDeviation));
		auto isInG = ((special_G_standard - specialColorDeviation) <= special_G) && (special_G <= (special_G_standard + specialColorDeviation));
		auto isInB = ((special_B_standard - specialColorDeviation) <= special_B) && (special_B <= (special_B_standard + specialColorDeviation));
		if ((!isInR) || (!isInG) || (!isInB))
		{
			_isbad = true;
			info.isDrawSpecialColor = true;
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_edgeDamage( ButtonDefectInfo& info)
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
				info.isDrawedgeDamage = true;
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

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_pore( ButtonDefectInfo& info)
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
				info.isDrawpore = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_paint( ButtonDefectInfo& info)
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
				info.isDrawpaint = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_brokenEye( ButtonDefectInfo& info)
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
				info.isDrawbrokenEye = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_crack( ButtonDefectInfo& info)
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
				info.isDrawcrack = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_grindStone( ButtonDefectInfo& info)
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
				info.isDrawgrindStone = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_blockEye( ButtonDefectInfo& info)
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
				info.isDrawblockEye = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_materialHead( ButtonDefectInfo& info)
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
				info.isDrawmaterialHead = true;
				break;
			}
		}
	}
}

void ImageProcessor::run_OpenRemoveFunc_process_defect_info_largeColor( ButtonDefectInfo& info)
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

		auto largeR = info.large_R;
		auto largeG = info.large_G;
		auto largeB = info.large_B;
		auto isInR = ((largeR - deviation)<= info.special_R) && (info.special_R<=(largeR + deviation));
		auto isInG = ((largeG - deviation) <= info.special_G) && (info.special_G <= (largeG + deviation));
		auto isInB = ((largeB - deviation) <= info.special_B) && (info.special_B <= (largeB + deviation));

		if ((!isInR) || (!isInG) || (!isInB))
		{
			_isbad = true;
			info.isDrawlargeColor = true;
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
	getCrackInfo(info, processResult, index[ClassId::liehen]);
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
	getCrackInfo(info, processResult, index[ClassId::liehen]);
	getBrokenEyeInfo(info, processResult, index[ClassId::poyan]);
	getPaintInfo(info, processResult, index[ClassId::mofa]);
	getLargeColorDifference(info, processResult, index, mat);
	getSpecialColorDifference(info, processResult, index, mat);
}

void ImageProcessor::getEliminationInfo_positive(ButtonDefectInfo& info,
	const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index,
	const cv::Mat& mat)
{
	auto& globalData = GlobalStructData::getInstance();
	if (globalData.mainWindowConfig.isPositive)
	{
		for (const auto &item : index[ClassIdPositive::Bad])
		{
			info.positive.push_back(processResult[item].score*100);
		}
	}
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
		processor->buildOnnxRuntimeOO(modelOnnxOOPath);
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

void ImageProcessingModule::reLoadOnnxOO()
{

	for (auto& item : _processors)
	{
		item->reloadOnnxRuntimeOO(modelOnnxOOPath);
	}

}

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

