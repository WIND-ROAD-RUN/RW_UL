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

		// 检查 frame 是否有效
		if (frame.image.empty()) {
			continue; // 跳过空帧
		}

		auto& globalData = GlobalStructDataSmartCroppingOfBags::getInstance();

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

void ImageProcessorSmartCroppingOfBags::run_debug(MatInfo& frame)
{
	emit imageReady(QPixmap::fromImage(rw::rqw::cvMatToQImage(frame.image)));
}

void ImageProcessorSmartCroppingOfBags::run_monitor(MatInfo& frame)
{

}

void ImageProcessorSmartCroppingOfBags::run_OpenRemoveFunc(MatInfo& frame)
{

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
			globalStruct.priorityQueue1->insert(frame.time, frame.time);
			break;
		case 2:
			globalStruct.priorityQueue2->insert(frame.time, frame.time);
			break;
		default:
			break;
		}
	}
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

void ImageProcessorSmartCroppingOfBags::getEliminationInfo_debug(SmartCroppingOfBagsDefectInfo& info,
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
	return false;
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
	textList.push_back(info.time);

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
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			shudangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			huapoText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			jietouText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			guasiText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			podongText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			zangwuText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			noshudangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			modianText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			loumoText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			xishudangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			erweimaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			damodianText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			kongdongText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			sebiaoText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			yinshuaquexianText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			xiaopodongText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
			jiaodaiText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
	textList.push_back(info.time);

	// 黑疤
	if (!info.heibaList.empty()) {
		QString queyaText("黑疤:");
		for (const auto& item : info.heibaList) {
			queyaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(queyaText);
	}
	// 疏档
	if (!info.shudangList.empty()) {
		QString shudangText("疏档:");
		for (const auto& item : info.shudangList) {
			shudangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(shudangText);
	}
	// 划破
	if (!info.huapoList.empty()) {
		QString huapoText("划破:");
		for (const auto& item : info.huapoList) {
			huapoText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(huapoText);
	}
	// 接头
	if (!info.jietouList.empty()) {
		QString jietouText("接头:");
		for (const auto& item : info.jietouList) {
			jietouText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(jietouText);
	}
	// 挂丝
	if (!info.guasiList.empty()) {
		QString guasiText("挂丝:");
		for (const auto& item : info.guasiList) {
			guasiText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(guasiText);
	}
	// 破洞
	if (!info.podongList.empty()) {
		QString guasiText("破洞:");
		for (const auto& item : info.podongList) {
			guasiText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(guasiText);
	}
	// 脏污
	if (!info.zangwuList.empty()) {
		QString zangwuText("脏污:");
		for (const auto& item : info.zangwuList) {
			zangwuText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(zangwuText);
	}
	// 无疏档
	if (!info.noshudangList.empty()) {
		QString noshudangText("无疏档:");
		for (const auto& item : info.noshudangList) {
			noshudangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(noshudangText);
	}
	// 墨点
	if (!info.modianList.empty()) {
		QString modianText("墨点:");
		for (const auto& item : info.modianList) {
			modianText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(modianText);
	}
	// 漏膜
	if (!info.loumoList.empty()) {
		QString loumoText("漏膜:");
		for (const auto& item : info.loumoList) {
			loumoText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(loumoText);
	}
	// 稀疏档
	if (!info.xishudangList.empty()) {
		QString xishudangText("稀疏档:");
		for (const auto& item : info.xishudangList) {
			xishudangText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(xishudangText);
	}
	// 二维码
	if (!info.erweimaList.empty()) {
		QString erweimaText("二维码:");
		for (const auto& item : info.erweimaList) {
			erweimaText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(erweimaText);
	}
	// 大墨点
	if (!info.damodianList.empty()) {
		QString damodianText("大墨点:");
		for (const auto& item : info.damodianList) {
			damodianText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(damodianText);
	}
	// 孔洞
	if (!info.kongdongList.empty()) {
		QString kongdongText("孔洞:");
		for (const auto& item : info.kongdongList) {
			kongdongText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(kongdongText);
	}
	// 色标
	if (!info.sebiaoList.empty()) {
		QString sebiaoText("色标:");
		for (const auto& item : info.sebiaoList) {
			sebiaoText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(sebiaoText);
	}
	// 印刷缺陷
	if (!info.yinshuaquexianList.empty()) {
		QString yinshuaquexianText("印刷缺陷:");
		for (const auto& item : info.yinshuaquexianList) {
			yinshuaquexianText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(yinshuaquexianText);
	}
	// 小破洞
	if (!info.xiaopodongList.empty()) {
		QString xiaopodongText("小破洞:");
		for (const auto& item : info.xiaopodongList) {
			xiaopodongText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
		}
		textList.push_back(xiaopodongText);
	}
	// 胶带
	if (!info.jiaodaiList.empty()) {
		QString jiaodaiText("胶带:");
		for (const auto& item : info.jiaodaiList) {
			jiaodaiText.append(QString(" %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2));
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
	config.shapeType = rw::rqw::ImagePainter::ShapeType::Rectangle;
	config.color = rw::rqw::ImagePainter::toQColor(rw::rqw::ImagePainter::BasicColor::Green);

	// 黑疤
	for (const auto& item : info.heibaList)
	{
		if (!item.isDraw)
		{
			auto& heibaItem = processResult[item.index];
			config.text = QString("黑疤 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, heibaItem, config);
		}
	}
	// 疏档
	for (const auto& item : info.shudangList)
	{
		if (!item.isDraw)
		{
			auto& shudangItem = processResult[item.index];
			config.text = QString("疏档 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, shudangItem, config);
		}
	}
	// 划破
	for (const auto& item : info.huapoList)
	{
		if (!item.isDraw)
		{
			auto& huapoItem = processResult[item.index];
			config.text = QString("划破 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, huapoItem, config);
		}
	}
	// 接头
	for (const auto& item : info.jietouList)
	{
		if (!item.isDraw)
		{
			auto& jietouItem = processResult[item.index];
			config.text = QString("接头 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jietouItem, config);
		}
	}
	// 挂丝
	for (const auto& item : info.guasiList)
	{
		if (!item.isDraw)
		{
			auto& guasiItem = processResult[item.index];
			config.text = QString("挂丝 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, guasiItem, config);
		}
	}
	// 破洞
	for (const auto& item : info.podongList)
	{
		if (!item.isDraw)
		{
			auto& podongItem = processResult[item.index];
			config.text = QString("破洞 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, podongItem, config);
		}
	}
	// 脏污
	for (const auto& item : info.zangwuList)
	{
		if (!item.isDraw)
		{
			auto& zangwuItem = processResult[item.index];
			config.text = QString("脏污 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, zangwuItem, config);
		}
	}
	// 无疏档
	for (const auto& item : info.noshudangList)
	{
		if (!item.isDraw)
		{
			auto& noshudangItem = processResult[item.index];
			config.text = QString("无疏档 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, noshudangItem, config);
		}
	}
	// 墨点
	for (const auto& item : info.modianList)
	{
		if (!item.isDraw)
		{
			auto& modianItem = processResult[item.index];
			config.text = QString("墨点 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, modianItem, config);
		}
	}
	// 漏膜
	for (const auto& item : info.loumoList)
	{
		if (!item.isDraw)
		{
			auto& loumoItem = processResult[item.index];
			config.text = QString("漏膜 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, loumoItem, config);
		}
	}
	// 稀疏档
	for (const auto& item : info.xishudangList)
	{
		if (!item.isDraw)
		{
			auto& xishudangItem = processResult[item.index];
			config.text = QString("稀疏档 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xishudangItem, config);
		}
	}
	// 二维码
	for (const auto& item : info.erweimaList)
	{
		if (!item.isDraw)
		{
			auto& erweimaItem = processResult[item.index];
			config.text = QString("二维码 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, erweimaItem, config);
		}
	}
	// 大墨点
	for (const auto& item : info.damodianList)
	{
		if (!item.isDraw)
		{
			auto& damodianItem = processResult[item.index];
			config.text = QString("大墨点 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, damodianItem, config);
		}
	}
	// 孔洞
	for (const auto& item : info.kongdongList)
	{
		if (!item.isDraw)
		{
			auto& kongdongItem = processResult[item.index];
			config.text = QString("孔洞 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, kongdongItem, config);
		}
	}
	// 色标
	for (const auto& item : info.sebiaoList)
	{
		if (!item.isDraw)
		{
			auto& sebiaoItem = processResult[item.index];
			config.text = QString("色标 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, sebiaoItem, config);
		}
	}
	// 印刷缺陷
	for (const auto& item : info.yinshuaquexianList)
	{
		if (!item.isDraw)
		{
			auto& yinshuaquexianItem = processResult[item.index];
			config.text = QString("印刷缺陷 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, yinshuaquexianItem, config);
		}
	}
	// 小破洞
	for (const auto& item : info.xiaopodongList)
	{
		if (!item.isDraw)
		{
			auto& xiaopodongItem = processResult[item.index];
			config.text = QString("小破洞 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xiaopodongItem, config);
		}
	}
	// 胶带
	for (const auto& item : info.jiaodaiList)
	{
		if (!item.isDraw)
		{
			auto& jiaodaiItem = processResult[item.index];
			config.text = QString("胶带 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
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

	// 黑疤
	for (const auto& item : info.heibaList)
	{
		if (item.isDraw)
		{
			auto& heibaItem = processResult[item.index];
			config.text = QString("黑疤 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, heibaItem, config);
		}
	}
	// 疏档
	for (const auto& item : info.shudangList)
	{
		if (item.isDraw)
		{
			auto& shudangItem = processResult[item.index];
			config.text = QString("疏档 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, shudangItem, config);
		}
	}
	// 划破
	for (const auto& item : info.huapoList)
	{
		if (item.isDraw)
		{
			auto& huapoItem = processResult[item.index];
			config.text = QString("划破 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, huapoItem, config);
		}
	}
	// 接头
	for (const auto& item : info.jietouList)
	{
		if (item.isDraw)
		{
			auto& jietouItem = processResult[item.index];
			config.text = QString("接头 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jietouItem, config);
		}
	}
	// 挂丝
	for (const auto& item : info.guasiList)
	{
		if (item.isDraw)
		{
			auto& guasiItem = processResult[item.index];
			config.text = QString("挂丝 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, guasiItem, config);
		}
	}
	// 破洞
	for (const auto& item : info.podongList)
	{
		if (item.isDraw)
		{
			auto& podongItem = processResult[item.index];
			config.text = QString("破洞 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, podongItem, config);
		}
	}
	// 脏污
	for (const auto& item : info.zangwuList)
	{
		if (item.isDraw)
		{
			auto& zangwuItem = processResult[item.index];
			config.text = QString("脏污 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, zangwuItem, config);
		}
	}
	// 无疏档
	for (const auto& item : info.noshudangList)
	{
		if (item.isDraw)
		{
			auto& noshudangItem = processResult[item.index];
			config.text = QString("无疏档 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, noshudangItem, config);
		}
	}
	// 墨点
	for (const auto& item : info.modianList)
	{
		if (item.isDraw)
		{
			auto& modianItem = processResult[item.index];
			config.text = QString("墨点 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, modianItem, config);
		}
	}
	// 漏膜
	for (const auto& item : info.loumoList)
	{
		if (item.isDraw)
		{
			auto& loumoItem = processResult[item.index];
			config.text = QString("漏膜 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, loumoItem, config);
		}
	}
	// 稀疏档
	for (const auto& item : info.xishudangList)
	{
		if (item.isDraw)
		{
			auto& xishudangItem = processResult[item.index];
			config.text = QString("稀疏档 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xishudangItem, config);
		}
	}
	// 二维码
	for (const auto& item : info.erweimaList)
	{
		if (item.isDraw)
		{
			auto& erweimaItem = processResult[item.index];
			config.text = QString("二维码 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, erweimaItem, config);
		}
	}
	// 大墨点
	for (const auto& item : info.damodianList)
	{
		if (item.isDraw)
		{
			auto& damodianItem = processResult[item.index];
			config.text = QString("大墨点 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, damodianItem, config);
		}
	}
	// 孔洞
	for (const auto& item : info.kongdongList)
	{
		if (item.isDraw)
		{
			auto& kongdongItem = processResult[item.index];
			config.text = QString("孔洞 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, kongdongItem, config);
		}
	}
	// 色标
	for (const auto& item : info.sebiaoList)
	{
		if (item.isDraw)
		{
			auto& sebiaoItem = processResult[item.index];
			config.text = QString("色标 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, sebiaoItem, config);
		}
	}
	// 印刷缺陷
	for (const auto& item : info.yinshuaquexianList)
	{
		if (item.isDraw)
		{
			auto& yinshuaquexianItem = processResult[item.index];
			config.text = QString("印刷缺陷 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, yinshuaquexianItem, config);
		}
	}
	// 小破洞
	for (const auto& item : info.xiaopodongList)
	{
		if (item.isDraw)
		{
			auto& xiaopodongItem = processResult[item.index];
			config.text = QString("小破洞 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, xiaopodongItem, config);
		}
	}
	// 胶带
	for (const auto& item : info.jiaodaiList)
	{
		if (item.isDraw)
		{
			auto& jiaodaiItem = processResult[item.index];
			config.text = QString("胶带 %1 %2").arg(item.score, 0, 'f', 2).arg(item.area, 0, 'f', 2);
			rw::rqw::ImagePainter::drawShapesOnSourceImg(image, jiaodaiItem, config);
		}
	}
}


void ImageProcessingModuleSmartCroppingOfBags::BuildModule()
{
	for (int i = 0; i < _numConsumers; ++i) {
		static size_t workIndexCount = 0;
		ImageProcessorSmartCroppingOfBags* processor = new ImageProcessorSmartCroppingOfBags(_queue, _mutex, _condition, workIndexCount, this);
		workIndexCount++;
		processor->buildSegModelEngine(modelEnginePath);
		processor->imageProcessingModuleIndex = index;
		connect(processor, &ImageProcessorSmartCroppingOfBags::imageReady, this, &ImageProcessingModuleSmartCroppingOfBags::imageReady, Qt::QueuedConnection);
		connect(processor, &ImageProcessorSmartCroppingOfBags::imageNGReady, this, &ImageProcessingModuleSmartCroppingOfBags::imageNGReady, Qt::QueuedConnection);
		_processors.push_back(processor);
		processor->start();
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

void ImageProcessingModuleSmartCroppingOfBags::onFrameCaptured(cv::Mat frame, size_t index)
{
	if (frame.empty()) {
		return; // 跳过空帧
	}

	QMutexLocker locker(&_mutex);
	MatInfo mat;
	mat.image = frame;
	mat.index = index;
	mat.time = std::chrono::system_clock::now();	// 获取拍照的时间点
	_queue.enqueue(mat);
	_condition.wakeOne();
}

