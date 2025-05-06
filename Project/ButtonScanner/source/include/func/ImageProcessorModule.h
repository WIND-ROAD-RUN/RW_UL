#pragma once

#include"ime_ModelEngineFactory.h"

#include"ButtonUtilty.h"

#include <QObject>
#include <QQueue>
#include <QMutex>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>
#include <vector>

struct ButtonDefectInfo
{
public:
	QString time{};
	double outsideDiameter{};
public:
	size_t holeCount{};
	std::vector<float> aperture{};
	std::vector<float> holeCentreDistance{};
public:
	float R{};
	float G{};
	float B{};
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

static void drawHole(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

static void drawBody(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

static std::vector<std::vector<size_t>> getAllIndexInMaxBody(const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, size_t deviationValue=10);

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
	void getEliminationInfo(ButtonDefectInfo & info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index);
	void getHoleInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getBodyInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);;

private:
	std::unique_ptr<rw::ModelEngine> _modelEngineOT;
public:
	void buildModelEngineOT(const QString& enginePath);
private:
	std::vector<std::vector<size_t>> filterEffectiveIndexes(std::vector<rw::DetectionRectangleInfo> info);
	void drawLine(QImage& image);
	void drawLine_locate(QImage& image, size_t locate);
	void drawVerticalBoundaryLine(QImage & image);
	void drawButtonDefectInfoText(QImage& image,const ButtonDefectInfo& info);
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
