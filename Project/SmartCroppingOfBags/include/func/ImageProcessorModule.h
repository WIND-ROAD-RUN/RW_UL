#pragma once

#include"ime_ModelEngineFactory.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <QPixmap>
#include <rqw_ImageSaveEngine.h>

#include"ImageCollage.hpp"
#include"dsl_TimeBasedCache.hpp"
#include"dsl_CacheFIFOThreadSafe.hpp"
#include"ImageProcessorUtilty.h"


class ImageProcessorSmartCroppingOfBags;

class ImageProcessingModuleSmartCroppingOfBags : public QObject {
	Q_OBJECT
public:
	QString modelEnginePath;
private:
	std::shared_ptr<ImageCollage> _imageCollage = nullptr;
	std::shared_ptr<rw::dsl::TimeBasedCache<Time, Time>> _historyTimes = nullptr;
	std::shared_ptr<rw::dsl::TimeBasedCache<Time, HistoryDetectInfo>> _historyResult = nullptr;
	std::shared_ptr<rw::dsl::CacheFIFOThreadSafe<Time, bool>> _timeBool = nullptr;
public:
	// 初始化图像处理模块
	void BuildModule();
	void setCollageImageNum(size_t num);
public:
	ImageProcessingModuleSmartCroppingOfBags(int numConsumers, QObject* parent = nullptr);

	~ImageProcessingModuleSmartCroppingOfBags();
public:
	cv::Mat mat1;
	cv::Mat mat2;
public slots:
	// 相机回调函数
	void onFrameCaptured(cv::Mat frame, size_t index);

signals:
	void imageReady(QPixmap image);
	void imageNGReady(QPixmap image, size_t index, bool isbad);
signals:
	void appendPixel(double pixel);
public:
	std::vector<ImageProcessorSmartCroppingOfBags*> getProcessors() const {
		return _processors;
	}

private:
	QQueue<MatInfo> _queue;
	QMutex _mutex;
	QWaitCondition _condition;
	std::vector<ImageProcessorSmartCroppingOfBags*> _processors;
	int _numConsumers;
public:
	size_t index;
};


class ImageProcessorSmartCroppingOfBags : public QThread
{
	Q_OBJECT
public:
	int imageProcessingModuleIndex;
public:
	std::shared_ptr<ImageCollage> _imageCollage = nullptr;
	//这个时间的长度，要向外提供接口，设置times数组的长度，从而决定了拼成的张数
	std::shared_ptr<rw::dsl::TimeBasedCache<Time, Time>> _historyTimes = nullptr;
	std::shared_ptr<rw::dsl::TimeBasedCache<Time, HistoryDetectInfo>> _historyResult = nullptr;
	std::shared_ptr<rw::dsl::CacheFIFOThreadSafe<Time, bool>> _timeBool = nullptr;
public:
	ImageProcessorSmartCroppingOfBags(QQueue<MatInfo>& queue,
		QMutex& mutex,
		QWaitCondition& condition,
		int workIndex,
		QObject* parent = nullptr);
private:
	Time _lastQieDaoTime{};
	Time _qieDaoTime{};
	bool _isQieDao{};
protected:
	void run() override;

private:
	void run_debug(MatInfo& frame);				// 不开剔废时候的调试模式

	void run_OpenRemoveFunc(MatInfo& frame);	// 开启剔废功能时的处理模式

private:
	void drawDebugTextInfoOnQImage(QImage & image,const HistoryDetectInfo &info);
public:
	std::vector<Time> getValidTime(const std::vector<Time>& times);
private:
	void getErrorLocation(const Time& times, const SmartCroppingOfBagsDefectInfo& info);
	void getErrorLocation(const std::vector<Time>& times);
	// AI模型处理拼接图像
	std::vector<rw::DetectionRectangleInfo> processCollageImage_debug(const cv::Mat& mat);
	// 获取上个时间戳的图像的高度
	int splitRecognitionBox_debug(const std::vector<Time>& time);
	//获取切刀线
	void getCutLine(const std::vector<Time>& time,const MatInfo& frame);
	// 将识别框分割成上一次图像的,与这一次图像的识别框,并重新添加到相应的识别框中
	void regularizedTwoRecognitionBox_debug(const int& previousMatHeight, const Time& previousTime, const Time& nowTime, std::vector<rw::DetectionRectangleInfo>& allDetectRec, const
	                                        QString& processTime);
	// 将属于上一张图像的识别框合并到上一次的图像识别框中
	void mergeCurrentProcessLastResultWithLastProcessResult_debug(const int& previousMatHeight, const Time& time, std::vector<rw::DetectionRectangleInfo>& allDetectRec);
	// 将属于当前图像的识别框重新计算Y轴并合并到当前图像识别框中
	void addCurrentResultToHistoryResult_debug(const int& previousMatHeight, std::vector<rw::DetectionRectangleInfo>& nowDetectRec, const Time& nowTime, const QString
	                                           & processTime);
	// 返回包含当前时间点的count个时间戳集合
	std::vector<Time> getCurrentWithBeforeFourTimes_debug(const Time& time, int count, bool isBefore = true,
		bool ascending = true);
	// 获得时间集合对应的图像
	void getUnprocessedSouceImage_debug(const std::vector<Time>& fiveTimes, std::vector<TimeFrameMatInfo>& images);
	// 对五张图像进行绘画检测框操作
	std::vector<TimeFrameQImageInfo> drawUnprocessedMatMaskInfo_debug(const std::vector<TimeFrameMatInfo>& fiveMats);
	void drawCutLine(TimeFrameQImageInfo& info);

	// 拼接绘画好的五张图像
	QPixmap collageMaskImage_debug(const QVector<QImage>& fiveQImages);
	// 随机添加五个检测框
	void getRandomDetecionRec_debug(const ImageCollage::CollageImage& collageImage, std::vector<rw::DetectionRectangleInfo>& detectionRec); // 获取随机的检测框

	QImage getCollageImage(const std::vector<TimeFrameQImageInfo>& infos);
private:
	void run_OpenRemoveFunc_process_defect_info(const Time& time);
	void run_OpenRemoveFunc_process_defect_info(SmartCroppingOfBagsDefectInfo& info);
	void run_OpenRemoveFunc_emitErrorInfo(const Time& time) const;
	void save_image(rw::rqw::ImageInfo& imageInfo, const QImage& image);
	void save_image_work(rw::rqw::ImageInfo& imageInfo, const QImage& image);

private:
	// 调试模式下将对应的缺陷信息添加到SmartCroppingOfBagsDefectInfo中
	void getEliminationInfo_debug(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index);
	// 剔废模式下将对应的缺陷信息添加到SmartCroppingOfBagsDefectInfo中
	void getEliminationInfo_defect(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index);

	static std::vector<std::vector<size_t>> getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info);

private:
	// AI模型参数
	std::unique_ptr<rw::ModelEngine> _modelEngine;
public:
	// 构建模型引擎
	void buildSegModelEngine(const QString& enginePath);		// Segmentation 模型

private:
	// 开启剔废时, 过滤出有效索引
	std::vector<std::vector<size_t>> filterEffectiveIndexes_defect(std::vector<rw::DetectionRectangleInfo> info);

public:
	// 开启剔废情况下绘制缺陷相关的信息(符合条件的缺陷会用红色显示)
	void drawSmartCroppingOfBagsDefectInfoText_defect(QImage& image, const SmartCroppingOfBagsDefectInfo& info);

public:
	// 绘画绿色的检测框
	void drawDefectRec_green(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex, const SmartCroppingOfBagsDefectInfo& info);
	// 绘画红色的检测框
	void drawDefectRec_red(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex, const SmartCroppingOfBagsDefectInfo& info);

private:
	// 判断是否有缺陷
	bool _isbad{ false };
private:
	size_t imagesCount{ 0 };
private:
	QQueue<MatInfo>& _queue;
	QMutex& _mutex;
	QWaitCondition& _condition;
	int _workIndex;

signals:
	void imageReady(QPixmap image);
	void imageNGReady(QPixmap image, size_t index, bool isbad);
signals:
	void appendPixel(double pixel);
};


