#pragma once

#include <QThread>
#include <QDebug>
#include <atomic>

#include"imeot_ModelEngineOT.h"

#include"opencv2/opencv.hpp"
#include<QProcess>

enum class ModelType
{
	Segment,
	ObjectDetection,
};

class AiTrainModule : public QThread
{
	Q_OBJECT
private:
	ModelType _modelType;
public:
	void setModelType(ModelType type) { _modelType = type; }
public:
	QProcess* _processTrainModel;
	QProcess* _processExportModel;
private:
	int _frameHeight;
	int _frameWidth;
public:
	using labelAndImg = QPair<QString, rw::imeot::ProcessRectanglesResultOT>;
	using DataItem = QPair<QString, QString>;
	explicit AiTrainModule(QObject* parent = nullptr);

	~AiTrainModule() override;

	void startTrain();

private:
	std::unique_ptr<rw::imeot::ModelEngineOT> labelEngine;
private:
	rw::imeot::ProcessRectanglesResultOT getBody(std::vector<rw::imeot::ProcessRectanglesResultOT>& processRectanglesResult, bool& hasBody);
	QVector<DataItem> getDataSet(const QVector<labelAndImg>& annotationDataSet, ModelType type, int classId);
	QVector<DataItem> getSegmentDataSet(const QVector<labelAndImg>& annotationDataSet, int classId);
	QVector<DataItem> getObjectDetectionDataSet(const QVector<labelAndImg>& annotationDataSet, int classId);
private:
	void clear_older_trainData();
	void copyTrainData(const QVector<AiTrainModule::DataItem>& dataSet);
	void copyTrainImgData(const QVector<AiTrainModule::DataItem>& dataSet, const QString& path);
	void copyTrainLabelData(const QVector<AiTrainModule::DataItem>& dataSet, const QString& path);
private:
	void trainSegmentModel();
	void trainObbModel();
	void exportOnnxModel();
	void copyModelToTemp();
	void packageModelToStorage();
private:
	void copy_all_files_to_storage(const QString& sourceFilePath, const QString& storage);
public:
	cv::Mat getMatFromPath(const QString& path);
protected:
	void run() override;
private:
	QVector<labelAndImg> annotation_data_set(bool isBad);
private:
	int parseProgressOO(const QString& logText, int& totalTasks);
	int parseProgressSO(const QString& logText, int& totalTasks);
signals:
	void appRunLog(QString log);
	void updateProgress(int value, int total);
	void updateTrainTitle(QString s);
	void updateTrainState(bool isTrain);
public slots:
	void handleTrainModelProcessOutput();
	void handleTrainModelProcessError();
	void handleTrainModelProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

	void handleExportModelProcessOutput();
	void handleExportModelProcessError();
	void handleExportModelProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	void cancelTrain();
};
