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
	bool isoutsideDiameter{ false };
public:
	size_t holeCount{};
	bool isDrawholeCount{ false };

	std::vector<float> aperture{};
	bool isDrawaperture{ false };

	std::vector<float> holeCentreDistance{};
	bool isDraweholeCentreDistance{ false };
public:
	float special_R{};
	float special_G{};
	float special_B{};
	bool isDrawSpecialColor{ false };
public:
	float large_R{};
	float large_G{};
	float large_B{};
	bool isDrawlargeColor{ false };
public:
	std::vector<double> edgeDamage;
	bool isDrawedgeDamage{ false };

	std::vector<double> pore;
	bool isDrawpore{ false };

	std::vector<double> paint;
	bool isDrawpaint{ false };

	std::vector<double> brokenEye;
	bool isDrawbrokenEye{ false };

	std::vector<double> crack;
	bool isDrawcrack{ false };

	std::vector<double> grindStone;
	bool isDrawgrindStone{ false };

	std::vector<double> blockEye;
	bool isDrawblockEye{ false };

	std::vector<double> materialHead;
	bool isDrawmaterialHead{ false };

public:

	std::vector<double> positive;
	bool isDrawpositiver{ false };
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

struct ImageProcessUtilty
{
	enum class CropMode {
		Rectangle,       // 计算矩形区域的平均 RGB 值
		InscribedCircle  // 计算矩形内接圆的平均 RGB 值
	};

	static cv::Vec3f calculateRegionRGB(const cv::Mat& image,
		const rw::DetectionRectangleInfo& total,
		CropMode mode,
		const std::vector<size_t>& index,
		const std::vector<rw::DetectionRectangleInfo>& processResult,
		CropMode excludeMode = CropMode::Rectangle);

	static cv::Vec3f calculateRegionRGB(const cv::Mat& image, const cv::Rect& rect, CropMode mode,
		std::vector<cv::Rect> excludeRegions = {}, CropMode excludeMode = CropMode::Rectangle);

	static std::vector<std::vector<size_t>> getClassIndex(const std::vector<rw::DetectionRectangleInfo>& info);

	static void drawHole(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

	static void drawBody(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& index);

	static std::vector<std::vector<size_t>> getAllIndexInMaxBody(const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, size_t deviationValue = 10);
};

struct MatInfo {
	cv::Mat image;
	float location;
	size_t index;
};

class ImageProcessor : public QThread {
	Q_OBJECT
private:
	std::vector<float> large_R_list{};
	std::vector<float> large_G_list{};
	std::vector<float> large_B_list{};
public:
	void clearLargeRGBList();
private:
	bool _isbad{ false };
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
private:
	void run_OpenRemoveFunc(MatInfo& frame);
	void run_OpenRemoveFunc_process_defect_info_positive(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_hole(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_body(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_specialColor(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_edgeDamage(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_emitErrorInfo(const MatInfo& frame) const;
	void run_OpenRemoveFunc_process_defect_info_pore(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_paint(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_brokenEye(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_crack(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_grindStone(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_blockEye(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_materialHead(ButtonDefectInfo& info);
	void run_OpenRemoveFunc_process_defect_info_largeColor(ButtonDefectInfo& info);
signals:
	void imageReady(QPixmap image);
private:
	void getEliminationInfo_debug(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);
	void getEliminationInfo_defect(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);
	void getEliminationInfo_positive(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const cv::Mat& mat);

	void getHoleInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getBodyInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getSpecialColorDifference(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const
		cv::Mat& mat);
	void getLargeColorDifference(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& index, const
		cv::Mat& mat);
	void getEdgeDamageInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getPoreInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getPaintInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getBrokenEyeInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getCrackInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getGrindStoneInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getBlockEyeInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void getMaterialHeadInfo(ButtonDefectInfo& info, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);

private:
	std::unique_ptr<rw::ModelEngine> _modelEngineOT;
	std::unique_ptr<rw::ModelEngine> _onnxRuntimeOO;
public:
	void buildModelEngineOT(const QString& enginePath);
	void buildOnnxRuntimeOO(const QString& enginePath);
public:
	void reloadOnnxRuntimeOO(const QString& enginePath);
private:
	std::vector<std::vector<size_t>> filterEffectiveIndexes_debug(std::vector<rw::DetectionRectangleInfo> info);
	std::vector<std::vector<size_t>> filterEffectiveIndexes_defect(std::vector<rw::DetectionRectangleInfo> info);
	std::vector<std::vector<size_t>> filterEffectiveIndexes_positive(std::vector<rw::DetectionRectangleInfo> info);

	std::vector<std::vector<size_t>>
		getIndexInBoundary
		(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index);

	bool isInBoundary(const rw::DetectionRectangleInfo& info);

	std::vector<std::vector<size_t>>
		getIndexInShieldingRange
		(const std::vector<rw::DetectionRectangleInfo>& info, const std::vector<std::vector<size_t>>& index) const;

	static bool isInShieldRange(const QPoint& outCentral, int outR, const QPoint& innerCentral, int innerR, const QPoint& point);
public:
	void drawButtonDefectInfoText_defect(QImage& image, const ButtonDefectInfo& info);
	void appendHolesCountDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendBodyCountDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendSpecialColorDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendLargeColorDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendEdgeDamageDefectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendPoreDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendPaintDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendBrokenEyeDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendPositiveDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendCrackDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendGrindStoneDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendBlockEyeDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
	void appendMaterialHeadDectInfo(QVector<QString>& textList, const ButtonDefectInfo& info);
public:
	void drawLine(QImage& image);
	void drawLine_locate(QImage& image, size_t locate);
	void drawVerticalBoundaryLine(QImage& image);
	void drawButtonDefectInfoText(QImage& image, const ButtonDefectInfo& info);
	void drawShieldingRange(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<size_t>& processIndex);
	void drawErrorRec(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex);
	void drawErrorRec_error(QImage& image, const std::vector<rw::DetectionRectangleInfo>& processResult, const std::vector<std::vector<size_t>>& processIndex);

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

	void clearLargeRGBList();
public:
	void reLoadOnnxOO();
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
