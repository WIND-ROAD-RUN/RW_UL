#pragma once

#include"ime_ModelEngineFactory.h"

#include <QObject>
#include <QQueue>
#include <QMutex>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>
#include <vector>
#include <QThread>
#include <QPixmap>
#include <rqw_ImageSaveEngine.h>

#include"ImageCollage.hpp"
#include"dsl_TimeBasedCache.hpp"
#include"dsl_CacheFIFOThreadSafe.hpp"


// 智能裁切吨袋检测信息
struct SmartCroppingOfBagsDefectInfo
{
public:
	// AI运行计时
	QString time;

public:
	// 缺陷
	struct DetectItem
	{
	public:
		double score = 0;	// 分数
		double area = 0;	// 面积
		int index = -1;		// 在processResult中的索引位置
		bool isDraw = false;	// 是否满足剔废条件绘画红框
	};

	std::vector<DetectItem> heibaList;         // 黑疤
	std::vector<DetectItem> shudangList;       // 疏档
	std::vector<DetectItem> huapoList;         // 划破
	std::vector<DetectItem> jietouList;        // 接头
	std::vector<DetectItem> guasiList;         // 挂丝
	std::vector<DetectItem> podongList;        // 破洞
	std::vector<DetectItem> zangwuList;        // 脏污
	std::vector<DetectItem> noshudangList;     // 无疏档
	std::vector<DetectItem> modianList;        // 墨点
	std::vector<DetectItem> loumoList;         // 漏膜
	std::vector<DetectItem> xishudangList;     // 稀疏档
	std::vector<DetectItem> erweimaList;       // 二维码
	std::vector<DetectItem> damodianList;      // 大墨点
	std::vector<DetectItem> kongdongList;      // 孔洞
	std::vector<DetectItem> sebiaoList;        // 色标
	std::vector<DetectItem> yinshuaquexianList;// 印刷缺陷
	std::vector<DetectItem> xiaopodongList;    // 小破洞
	std::vector<DetectItem> jiaodaiList;       // 胶带


};

// 图片画图模块
struct ImagePainter
{
	enum Color {
		White,
		Red,
		Green,
		Blue,
		Yellow,
		Cyan,
		Magenta,
		Black
	};

	static QColor ColorToQColor(Color c);

	static void drawTextOnImage(QImage& image, const QVector<QString>& texts, const QVector<Color>& colorList = { Color::Red,Color::Green }, double proportion = 0.8);
};

// 图片信息
struct MatInfo {
	rw::rqw::ElementInfo<cv::Mat> image;
	size_t index{};	// 拍照的相机的下标
	Time time;
	double location{};
public:
	// 默认构造函数
	MatInfo() = default;

	// 参数化构造函数
	MatInfo(const rw::rqw::ElementInfo<cv::Mat>& element) : image(element) {}

	// 拷贝构造函数
	MatInfo(const MatInfo& other)
		: image(other.image), index(other.index), time(other.time) {
	}

	// 拷贝赋值运算符（可选）
	MatInfo& operator=(const MatInfo& other) {
		if (this != &other) {
			image = other.image;
			index = other.index;
			time = other.time;
			location = other.location;
		}
		return *this;
	}
};

struct HistoryDetectInfo
{
	std::vector<rw::DetectionRectangleInfo> processResult;
public:
	HistoryDetectInfo() = default;
	HistoryDetectInfo(const std::vector<rw::DetectionRectangleInfo>& result) : processResult(result) {}
	// 拷贝构造函数
	HistoryDetectInfo(const HistoryDetectInfo& other) : processResult(other.processResult) {}
	// 拷贝赋值运算符
	HistoryDetectInfo& operator=(const HistoryDetectInfo& other) {
		if (this != &other) {
			processResult = other.processResult;
		}
		return *this;
	}
};

class ImageProcessorSmartCroppingOfBags;

class ImageProcessingModuleSmartCroppingOfBags : public QObject {
	Q_OBJECT
public:
	QString modelEnginePath;
private:
	std::shared_ptr<ImageCollage> _imageCollage = nullptr;
	//这个时间的长度，要向外提供接口，设置times数组的长度，从而决定了拼成的张数
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
	std::shared_ptr<ImageCollage> _imageCollage = nullptr;
	//这个时间的长度，要向外提供接口，设置times数组的长度，从而决定了拼成的张数
	std::shared_ptr<rw::dsl::TimeBasedCache<Time, Time>> _historyTimes = nullptr;

	std::shared_ptr<rw::dsl::TimeBasedCache<Time, HistoryDetectInfo>> _historyResult = nullptr;
	std::shared_ptr<rw::dsl::CacheFIFOThreadSafe<Time, bool>> _timeBool = nullptr;
	size_t collageImagesNum = 5;
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
	std::vector<Time> getValidTime(const std::vector<Time>& times);


	void run_monitor(MatInfo& frame);			// 单纯的显示模式


private:
	// 调试模式用的封装函数
	// 获得当前图像的时间戳与前count张图像的时间戳的集合
	std::vector<Time> getTimesWithCurrentTime_debug(const Time& time, int count, bool isBefore = true,
		bool ascending = true);
	// 获取一个时间集合拼接而成的图像
	ImageCollage::CollageImage getCurrentWithBeforeTimeCollageTime_debug(const std::vector<Time>& times);
	// AI模型处理拼接图像
	std::vector<rw::DetectionRectangleInfo> processCollageImage_debug(const cv::Mat& mat);
	// 获取上个时间戳的图像的高度
	int splitRecognitionBox_debug(const std::vector<Time>& time);
	// 将识别框分割成上一次图像的,与这一次图像的识别框,并重新添加到相应的识别框中
	void regularizedTwoRecognitionBox_debug(const int& previousMatHeight, const Time& previousTime, const Time& nowTime, std::vector<rw::DetectionRectangleInfo>& allDetectRec);
	// 将属于上一张图像的识别框合并到上一次的图像识别框中
	void mergeCurrentProcessLastResultWithLastProcessResult_debug(const int& previousMatHeight, const Time& time, std::vector<rw::DetectionRectangleInfo>& allDetectRec);
	// 将属于当前图像的识别框重新计算Y轴并合并到当前图像识别框中
	void addCurrentResultToHistoryResult_debug(const int& previousMatHeight, std::vector<rw::DetectionRectangleInfo>& nowDetectRec, const Time& nowTime);
	// 返回包含当前时间点的count个时间戳集合
	std::vector<Time> getCurrentWithBeforeFourTimes_debug(const Time& time, int count, bool isBefore = true,
		bool ascending = true);
	// 获得时间集合对应的图像
	void getUnprocessedSouceImage_debug(std::vector<Time> fiveTimes, std::vector<cv::Mat>& images);
	// 获得五次时间集合对应的五张图像的检测结果
	void getUnprocessedHistoyProcessResult_debug(const Time& time, int count, std::vector<std::vector<rw::DetectionRectangleInfo>>& detectRecs,
		bool isBefore = true, bool ascending = true);
	// 对五张图像进行绘画检测框操作
	QVector<QImage> drawUnprocessedMatMaskInfo_debug(const std::vector<cv::Mat>& fiveMats, const std::vector<std::vector<rw::DetectionRectangleInfo>>& fiveMatDetects);
	// 拼接绘画好的五张图像
	QPixmap collageMaskImage_debug(const QVector<QImage>& fiveQImages);
	// 随机添加五个检测框
	void getRandomDetecionRec_debug(const ImageCollage::CollageImage& collageImage, std::vector<rw::DetectionRectangleInfo>& detectionRec); // 获取随机的检测框

private:
	// 剔废模式下的处理函数
	// 获得当前图像的时间戳与前count张图像的时间戳的集合
	std::vector<Time> getTimesWithCurrentTime_Defect(const Time& time, int count, bool isBefore = true,
		bool ascending = true);
	// 获取一个时间集合拼接而成的图像
	ImageCollage::CollageImage getCurrentWithBeforeTimeCollageTime_Defect(const std::vector<Time>& times);
	// AI模型处理拼接图像
	std::vector<rw::DetectionRectangleInfo> processCollageImage_Defect(const cv::Mat& mat);
	// 获取上个时间戳的图像的高度
	int splitRecognitionBox_Defect(const std::vector<Time>& time);
	// 将识别框分割成上一次图像的,与这一次图像的识别框,并重新添加到相应的识别框中
	void regularizedTwoRecognitionBox_Defect(const int& previousMatHeight, const Time& previousTime, const Time& nowTime, std::vector<rw::DetectionRectangleInfo>& allDetectRec);
	// 将属于上一张图像的识别框合并到上一次的图像识别框中
	void mergeCurrentProcessLastResultWithLastProcessResult_Defect(const int& previousMatHeight, const Time& time, std::vector<rw::DetectionRectangleInfo>& allDetectRec);
	// 将属于当前图像的识别框重新计算Y轴并合并到当前图像识别框中
	void addCurrentResultToHistoryResult_Defect(const int& previousMatHeight, std::vector<rw::DetectionRectangleInfo>& nowDetectRec, const Time& nowTime);


private:
	void run_OpenRemoveFunc(MatInfo& frame);	// 开启剔废功能时的处理模式
	// 处理拉链缺陷信息
	void run_OpenRemoveFunc_process_defect_info(SmartCroppingOfBagsDefectInfo& info);
	// 处理黑疤
	void run_OpenRemoveFunc_process_defect_info_Heiba(SmartCroppingOfBagsDefectInfo& info);
	// 处理疏档
	void run_OpenRemoveFunc_process_defect_info_Shudang(SmartCroppingOfBagsDefectInfo& info);
	// 处理划破
	void run_OpenRemoveFunc_process_defect_info_Huapo(SmartCroppingOfBagsDefectInfo& info);
	// 处理接头
	void run_OpenRemoveFunc_process_defect_info_Jietou(SmartCroppingOfBagsDefectInfo& info);
	// 处理挂丝
	void run_OpenRemoveFunc_process_defect_info_Guasi(SmartCroppingOfBagsDefectInfo& info);
	// 处理破洞
	void run_OpenRemoveFunc_process_defect_info_Podong(SmartCroppingOfBagsDefectInfo& info);
	// 处理脏污
	void run_OpenRemoveFunc_process_defect_info_Zangwu(SmartCroppingOfBagsDefectInfo& info);
	// 处理无疏档
	void run_OpenRemoveFunc_process_defect_info_Noshudang(SmartCroppingOfBagsDefectInfo& info);
	// 处理墨点
	void run_OpenRemoveFunc_process_defect_info_Modian(SmartCroppingOfBagsDefectInfo& info);
	// 处理漏膜
	void run_OpenRemoveFunc_process_defect_info_Loumo(SmartCroppingOfBagsDefectInfo& info);
	// 处理稀疏档
	void run_OpenRemoveFunc_process_defect_info_Xishudang(SmartCroppingOfBagsDefectInfo& info);
	// 处理二维码
	void run_OpenRemoveFunc_process_defect_info_Erweima(SmartCroppingOfBagsDefectInfo& info);
	// 处理大墨点
	void run_OpenRemoveFunc_process_defect_info_Damodian(SmartCroppingOfBagsDefectInfo& info);
	// 处理孔洞
	void run_OpenRemoveFunc_process_defect_info_Kongdong(SmartCroppingOfBagsDefectInfo& info);
	// 处理色标
	void run_OpenRemoveFunc_process_defect_info_Sebiao(SmartCroppingOfBagsDefectInfo& info);
	// 处理印刷缺陷
	void run_OpenRemoveFunc_process_defect_info_Yinshuaquexian(SmartCroppingOfBagsDefectInfo& info);
	// 处理小破洞
	void run_OpenRemoveFunc_process_defect_info_Xiaopodong(SmartCroppingOfBagsDefectInfo& info);
	// 处理胶带
	void run_OpenRemoveFunc_process_defect_info_Jiaodai(SmartCroppingOfBagsDefectInfo& info);
	// 检测到缺陷后发出错误信息
	void run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const;

	// 存图
	void save_image(rw::rqw::ImageInfo& imageInfo, const QImage& image);
	void save_image_work(rw::rqw::ImageInfo& imageInfo, const QImage& image);

	//监控IO
	void monitorIO();

signals:
	void imageReady(QPixmap image);
	void imageNGReady(QPixmap image, size_t index, bool isbad);
signals:
	void appendPixel(double pixel);

private:
	// 调试模式下将对应的缺陷信息添加到SmartCroppingOfBagsDefectInfo中
	void getEliminationInfo_debug(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);
	// 剔废模式下将对应的缺陷信息添加到SmartCroppingOfBagsDefectInfo中
	void getEliminationInfo_defect(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);

	// 抓取黑疤信息
	void getHeibaInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取疏档信息
	void getShudangInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取划破信息
	void getHuapoInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取接头信息
	void getJietouInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取挂丝信息
	void getGuasiInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取破洞信息
	void getPodongInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取脏污信息
	void getZangwuInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取无疏档信息
	void getNoshudangInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取墨点信息
	void getModianInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取漏膜信息
	void getLoumoInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取稀疏档信息
	void getXishudangInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取二维码信息
	void getErweimaInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取大墨点信息
	void getDamodianInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取孔洞信息
	void getKongdongInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取色标信息
	void getSebiaoInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取印刷缺陷信息
	void getYinshuaquexianInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取小破洞信息
	void getXiaopodongInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取胶带信息
	void getJiaodaiInfo(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);

	static std::vector<std::vector<size_t>> getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info);

private:
	// AI模型参数
	std::unique_ptr<rw::ModelEngine> _modelEngine;
public:
	// 构建模型引擎
	void buildSegModelEngine(const QString& enginePath);		// Segmentation 模型

private:
	// 不开启剔废时, 过滤出有效索引
	std::vector<std::vector<size_t>> filterEffectiveIndexes_debug(std::vector<rw::DetectionRectangleInfo> info);
	// 开启剔废时, 过滤出有效索引
	std::vector<std::vector<size_t>> filterEffectiveIndexes_defect(std::vector<rw::DetectionRectangleInfo> info);

	// 筛选出在上下左右限位内的缺陷的下标
	std::vector<std::vector<size_t>> getIndexInBoundary(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index);
	// 判断是否在上下左右限位内
	bool isInBoundary(const rw::DetectionRectangleInfo& info);


public:
	// 开启剔废情况下绘制缺陷相关的信息(符合条件的缺陷会用红色显示)
	void drawSmartCroppingOfBagsDefectInfoText_defect(QImage& image, const SmartCroppingOfBagsDefectInfo& info);
	// 添加各个缺陷信息到文本列表中
	void appendHeibaDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendShudangDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendHuapoDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendJietouDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendGuasiDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendPodongDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendZangwuDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendNoshudangDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendModianDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendLoumoDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendXishudangDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendErweimaDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendDamodianDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendKongdongDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendSebiaoDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendYinshuaquexianDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendXiaopodongDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);
	void appendJiaodaiDectInfo(QVector<QString>& textList, const SmartCroppingOfBagsDefectInfo& info);


public:
	// 在指定位置画竖线
	void drawVerticalLine_locate(QImage& image, size_t locate);
	// 画切刀线与屏蔽线
	void drawBoundariesLines(QImage& image);
	// 开启调试情况下绘制缺陷相关的信息
	void drawSmartCroppingOfBagsDefectInfoText_Debug(QImage& image, const SmartCroppingOfBagsDefectInfo& info);
	// 绘画绿色的检测框
	void drawDefectRec(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex, const SmartCroppingOfBagsDefectInfo& info);
	// 绘画红色的检测框
	void drawDefectRec_error(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex, const SmartCroppingOfBagsDefectInfo& info);

private:
	// 判断是否有缺陷
	bool _isbad{ false };

public:
	void setCollageImageNum(size_t num);
private:
	size_t _collageNum{ 5 };
	size_t imagesCount{ 0 };
private:
	QQueue<MatInfo>& _queue;
	QMutex& _mutex;
	QWaitCondition& _condition;
	int _workIndex;

public:
	int imageProcessingModuleIndex;

};


