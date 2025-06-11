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
	cv::Mat image;	// 图片内容
	std::chrono::system_clock::time_point time;	// 记录拍照瞬间的时间点
	size_t index;	// 拍照的相机的下标
};


class ImageProcessorSmartCroppingOfBags : public QThread
{
	Q_OBJECT

public:
	ImageProcessorSmartCroppingOfBags(QQueue<MatInfo>& queue,
		QMutex& mutex,
		QWaitCondition& condition,
		int workIndex,
		QObject* parent = nullptr);

protected:
	void run() override;

private:
	void run_debug(MatInfo& frame);				// 不开剔废时候的调试模式
	void run_monitor(MatInfo& frame);			// 单纯的显示模式

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

signals:
	void imageReady(QPixmap image);

private:
	// 调试模式下将对应的缺陷信息添加到ZipperDefectInfo中
	void getEliminationInfo_debug(SmartCroppingOfBagsDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);
	// 剔废模式下将对应的缺陷信息添加到ZipperDefectInfo中
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
	// 在指定位置画横线
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

private:
	QQueue<MatInfo>& _queue;
	QMutex& _mutex;
	QWaitCondition& _condition;
	int _workIndex;

public:
	int imageProcessingModuleIndex;

};


class ImageProcessingModuleZipper : public QObject {
	Q_OBJECT
public:
	QString modelEnginePath;
public:
	// 初始化图像处理模块
	void BuildModule();
public:
	ImageProcessingModuleZipper(int numConsumers, QObject* parent = nullptr);

	~ImageProcessingModuleZipper();

public slots:
	// 相机回调函数
	void onFrameCaptured(cv::Mat frame, size_t index);

signals:
	void imageReady(QPixmap image);

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