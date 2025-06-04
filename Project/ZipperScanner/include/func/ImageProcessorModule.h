#pragma once

#include"ime_ModelEngineFactory.h"

#include <QObject>
#include <QQueue>
#include <QMutex>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>
#include <vector>
#include<QThread>


// 拉链检测信息
struct ZipperDefectInfo
{
public:
	// 计时
	QString time;

public:
	// 缺陷
	struct DetectItem
	{
	public:
		double score;	// 分数
		double area;	// 面积
	};

	std::vector<DetectItem> queYaList;		// 缺牙
	std::vector<DetectItem> tangShangList;	// 烫伤
	std::vector<DetectItem> zangWuList;		// 脏污

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
	QString time;	// 记录拍照瞬间的时间点
	size_t index;	// 拍照的相机的下标
};


class ImageProcessor : public QThread 
{
	Q_OBJECT

public:
	ImageProcessor(QQueue<MatInfo>& queue,
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
	void run_OpenRemoveFunc_process_defect_info(ZipperDefectInfo& info);
	// 处理缺牙
	void run_OpenRemoveFunc_process_defect_info_QueYa(ZipperDefectInfo& info);
	// 处理烫伤
	void run_OpenRemoveFunc_process_defect_info_TangShang(ZipperDefectInfo& info);
	// 处理脏污
	void run_OpenRemoveFunc_process_defect_info_ZangWu(ZipperDefectInfo& info);
	// 检测到缺陷后发出错误信息
	void run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const;

signals:
	void imageReady(QPixmap image);

private:
	// 展示和分析图像的所有检测信息
	void getEliminationInfo_debug(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);
	// 判断产品是否需要剔除，并驱动后续的剔除动作或统计
	void getEliminationInfo_defect(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);

	// 抓取缺牙信息
	void getQueyaInfo(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取烫伤信息
	void getTangshangInfo(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	// 抓取脏污信息
	void getZangwuInfo(ZipperDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);

private:
	// AI模型参数
	std::unique_ptr<rw::ModelEngine> _modelEngineOT;
	std::unique_ptr<rw::ModelEngine> _onnxRuntimeOO;
public:
	// 构建模型引擎
	void buildModelEngineOT(const QString& enginePath);
	void buildOnnxRuntimeOO(const QString& enginePath);

private:
	// 不开启剔废时, 过滤出有效索引
	std::vector<std::vector<size_t>> filterEffectiveIndexes_debug(std::vector<rw::DetectionRectangleInfo> info);
	// 开启剔废时, 过滤出有效索引
	std::vector<std::vector<size_t>> filterEffectiveIndexes_defect(std::vector<rw::DetectionRectangleInfo> info);

	// 筛选出在上下左右限位内的缺陷
	std::vector<std::vector<size_t>> getIndexInBoundary(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index);
	bool isInBoundary(const rw::DetectionRectangleInfo& info);


public:
	// 绘制缺陷相关的信息(符合条件的缺陷会用红色显示)
	void drawZipperDefectInfoText_defect(QImage& image, const ZipperDefectInfo& info);
	// 添加各个缺陷信息到文本列表中
	void appendQueyaDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info);
	void appendTangshangDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info);
	void appendZangwuDectInfo(QVector<QString>& textList, const ZipperDefectInfo& info);


public:
	// 在指定位置画竖线
	void drawVerticalLine_locate(QImage& image, size_t locate);
	// 在指定位置画横线
	void drawHorizontalLine_locate(QImage& image, size_t locate);
	// 绘制缺陷相关的信息
	void drawZipperDefectInfoText(QImage& image, const ZipperDefectInfo& info);
	// 绘画绿色的检测框
	void drawErrorRec(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex);
	// 绘画红色的检测框
	void drawErrorRec_error(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex, const ZipperDefectInfo& info);

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


class ImageProcessingModule : public QObject {
	Q_OBJECT
public:
	QString modelEngineOTPath;
	QString modelNamePath;
	QString modelOnnxOOPath;
	QString modelOnnxSOPath;
public:
	// 初始化图像处理模块
	void BuildModule();
public:
	// 重新加载 Onnx 推理模型
	void reLoadOnnxOO();
public:
	ImageProcessingModule(int numConsumers, QObject* parent = nullptr);

	~ImageProcessingModule();

public slots:
	// 相机回调函数
	void onFrameCaptured(cv::Mat frame, float location, size_t index);

signals:
	void imageReady(QPixmap image);

	// 推送新图片
	void imgForDlgNewProduction(cv::Mat mat, size_t index);
public:
	std::vector<ImageProcessor*> getProcessors() const {
		return _processors;
	}

private:
	QQueue<MatInfo> _queue;
	QMutex _mutex;
	QWaitCondition _condition;
	std::vector<ImageProcessor*> _processors;
	int _numConsumers;
public:
	size_t index;
};


