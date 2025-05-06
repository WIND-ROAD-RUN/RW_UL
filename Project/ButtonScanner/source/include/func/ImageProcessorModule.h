#pragma once

#include"ime_ModelEngineFactory.h"

#include <QObject>
#include <QQueue>
#include <QMutex>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>
#include <vector>

//#include"imeoo_ModelEngineOO.h"
//#include"imest_ModelEngineST.h"
//#include"imeso_ModelEngineSO.h"
//#include"imeot_ModelEngineOT.h"
//#include "imeso_ModelEngineSO.h"

struct ClassId
{
	static const int Body = 1;
	static const int Hole = 0;
	static const int pobian = 2;
	static const int qikong = 3;
	static const int duyan = 4;
	static const int moshi = 5;
	static const int liaotou = 6;
	static const int zangwu = 7;
	static const int pokong = 8;
	static const int poyan = 9;
	static const int xiaoqikong = 10;
	static const int mofa = 11;
	static const int xiaopobian = 12;
	static const int baibian = 13;

};

struct ButtonInfo
{

};

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

static std::vector<std::vector<size_t>> getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info);

static void drawHole(cv::Mat& mat, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

static void drawBody(cv::Mat& mat, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

static std::vector<std::vector<size_t>> getAllIndexInMaxBody(const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

struct ImageProcessUtilty 
{
	enum class CropMode {
		Rectangle,       // 计算矩形区域的平均 RGB 值
		InscribedCircle  // 计算矩形内接圆的平均 RGB 值
	};

	static cv::Vec3f calculateRegionRGB(const cv::Mat& image, const cv::Rect& rect, CropMode mode,
		std::vector<cv::Rect> excludeRegions = {}, CropMode excludeMode = CropMode::Rectangle);
};

struct MatInfo {
	cv::Mat image;
	float location;
	size_t index;
};

class ImageProcessor : public QThread {
	Q_OBJECT

private:
	bool _isbad{ false };
private:
	std::vector<std::vector<size_t>>
	getIndexInBoundary
	(const std::vector<rw::DetectionRectangleInfo>& info,const std::vector<std::vector<size_t>> & index);

	bool isInBoundary(const rw::DetectionRectangleInfo & info);
public:
	ImageProcessor(QQueue<MatInfo>& queue,
		QMutex& mutex,
		QWaitCondition& condition,
		int workIndex,
		QObject* parent = nullptr);
protected:
	void run() override;
private:
	void run_debug(MatInfo& frame);
	void run_monitor(MatInfo& frame);
	void run_OpenRemoveFunc(MatInfo& frame);
signals:
	void imageReady(QPixmap image);

private:
	//std::unique_ptr<rw::imeot::ModelEngineOT> _modelEnginePtr;
	//std::unique_ptr<rw::imeoo::ModelEngineOO> _modelEnginePtrOnnxOO;
	//std::unique_ptr<rw::imeso::ModelEngineSO> _modelEnginePtrOnnxSO;
	std::unique_ptr<rw::ModelEngine> _modelEngineOT;
public:
	void buildModelEngineOT(const QString& enginePath);

	//void buildModelEngineOnnxOO(const QString& enginePath, const QString& namePath);
	//void buildModelEngineOnnxSO(const QString& enginePath, const QString& namePath);

private:
	bool isInArea(int x);
	//std::vector<rw::imeot::ProcessRectanglesResultOT> getDefectInBody(rw::imeot::ProcessRectanglesResultOT body, const std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResult);

private:
	/*cv::Mat processAI(MatInfo& frame, QVector<QString>& errorInfo, std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResult, std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResultTarget);

	rw::imeot::ProcessRectanglesResultOT getBody(std::vector<rw::imeot::ProcessRectanglesResultOT>& processRectanglesResult, bool& hasBody);
	rw::imeoo::ProcessRectanglesResultOO getBody(std::vector<rw::imeoo::ProcessRectanglesResultOO>& processRectanglesResult, bool& hasBody);
	rw::imeso::ProcessRectanglesResultSO getBody(std::vector<rw::imeso::ProcessRectanglesResultSO>& processRectanglesResult, bool& hasBody);

	void eliminationLogic(
		MatInfo& frame,
		cv::Mat& resultImage,
		QVector<QString>& errorInfo,
		std::vector<rw::imeot::ProcessRectanglesResultOT>& processRectanglesResult,
		std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResultTarget,
		std::vector<rw::imeoo::ProcessRectanglesResultOO>& processRectanglesResultOO,
		std::vector<rw::imeso::ProcessRectanglesResultSO>& processRectanglesResultSO);

	void drawErrorLocate(QImage& image, std::vector<rw::imeot::ProcessRectanglesResultOT>& vecRecogResult, const QColor& drawColor);*/

	void drawLine(QImage& image);
	void drawLine_locate(QImage& image, size_t locate);

	void drawVerticalBoundaryLine(cv::Mat & mat);

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
	void BuildModule();

	//void reloadOOModel();
	//void reloadSOModel();
public:
	ImageProcessingModule(int numConsumers, QObject* parent = nullptr);

	~ImageProcessingModule();

public slots:
	void onFrameCaptured(cv::Mat frame, float location, size_t index);

signals:
	void imageReady(QPixmap image);

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
