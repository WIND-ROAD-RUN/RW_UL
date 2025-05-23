#pragma once

#include <QThread>
#include <QDebug>
#include <atomic>

#include"ime_ModelEngineFactory.h"
#include "ImageProcessorModule.h"

#include"cdm_AiModelConfig.h"

#include"opencv2/opencv.hpp"
#include<QProcess>

enum class ModelType
{
	Color,
	BladeShape,
};

class AiTrainModule : public QThread
{
	Q_OBJECT
private:
	ModelType _modelType;
public:
	void setModelType(ModelType type) { _modelType = type; }
public:
	QProcess* _processTrainModelBladeShape{ nullptr };
	QProcess* _processExportToEngine{ nullptr };
public:
	QProcess* _processTrainModelColor1{ nullptr };
	QProcess* _processTrainModelColor2{ nullptr };
	QProcess* _processTrainModelColor3{ nullptr };
	QProcess* _processTrainModelColor4{ nullptr };
public:
	QProcess* _processExportToEngineColor1{ nullptr };
	QProcess* _processExportToEngineColor2{ nullptr };
	QProcess* _processExportToEngineColor3{ nullptr };
	QProcess* _processExportToEngineColor4{ nullptr };
private:
	int _frameHeight;
	int _frameWidth;
public:
	using labelAndImg = QPair<QString, rw::DetectionRectangleInfo>;
	using DataItem = QPair<QString, QString>;
	explicit AiTrainModule(QObject* parent = nullptr);

	~AiTrainModule() override;

	void startTrain();
private:
	rw::cdm::AiModelConfig config;
private:
	void iniConfig();
private:
	std::unique_ptr<rw::ModelEngine> labelEngine;
private:
	QVector<DataItem> getDataSet(const QVector<labelAndImg>& annotationDataSet, int classId);
	QVector<DataItem> getSegmentDataSet(const QVector<labelAndImg>& annotationDataSet, int classId);
	QVector<DataItem> getObjectDetectionDataSet(const QVector<labelAndImg>& annotationDataSet, int classId);
private:
	void clear_older_trainData();
	void clear_older_trainData_color();
	void copyTrainData(const QVector<AiTrainModule::DataItem>& dataSet);
	void copyTrainImgData(const QVector<AiTrainModule::DataItem>& dataSet, const QString& path);
	void copyTrainLabelData(const QVector<AiTrainModule::DataItem>& dataSet, const QString& path);
private:
	void trainColorModel(int index);
	void trainShapeModel();
	void copyModelToTemp();
	void packageModelToStorage();
	void copyModelToTempColor(int workIndex);
private:
	void copy_all_files_to_storage(const QString& sourceFilePath, const QString& storage);
public:
	cv::Mat getMatFromPath(const QString& path);
protected:
	void run() override;
private:
	QVector<labelAndImg> annotation_data_set_bladeShape(bool isBad);
	QVector<labelAndImg> annotation_data_set_color(bool isBad, int workIndex);
private:
	void exportModelToEngine();
private:
	void exportColor1ModelToEngine();
	void exportColor2ModelToEngine();
	void exportColor3ModelToEngine();
	void exportColor4ModelToEngine();
private:
	int parseProgressOO(const QString& logText, int& totalTasks);
	int parseProgressSO(const QString& logText, int& totalTasks);

private:
	QVector<AiTrainModule::DataItem> dataSetGood1;
	QVector<AiTrainModule::DataItem> dataSetGood2;
	QVector<AiTrainModule::DataItem> dataSetGood3;
	QVector<AiTrainModule::DataItem> dataSetGood4;

	QVector<AiTrainModule::DataItem> dataSetBad1;
	QVector<AiTrainModule::DataItem> dataSetBad2;
	QVector<AiTrainModule::DataItem> dataSetBad3;
	QVector<AiTrainModule::DataItem> dataSetBad4;
signals:
	void appRunLog(QString log);
	void updateProgress(int value, int total);
	void updateTrainTitle(QString s);
	void updateTrainState(bool isTrain);
public slots:
	//BladeShapeTrain
	void handleProcessTrainModelBladeShapeOutput();
	void handleProcessTrainModelBladeShapeError();
	void handleProcessTrainModelBladeShapeFinished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//BladeShapeExport
	void handleProcessExportEngineBladeShapeOutput();
	void handleProcessExportEngineBladeShapeError();
	void handleProcessExportEngineBladeShapeFinished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color1Train
	void handleProcessTrainModelColor1Output();
	void handleProcessTrainModelColor1Error();
	void handleProcessTrainModelColor1Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color2Train
	void handleProcessTrainModelColor2Output();
	void handleProcessTrainModelColor2Error();
	void handleProcessTrainModelColor2Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color3Train
	void handleProcessTrainModelColor3Output();
	void handleProcessTrainModelColor3Error();
	void handleProcessTrainModelColor3Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color4Train
	void handleProcessTrainModelColor4Output();
	void handleProcessTrainModelColor4Error();
	void handleProcessTrainModelColor4Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color1Export
	void handleProcessExportEngineColor1Output();
	void handleProcessExportEngineColor1Error();
	void handleProcessExportEngineColor1Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color2Export
	void handleProcessExportEngineColor2Output();
	void handleProcessExportEngineColor2Error();
	void handleProcessExportEngineColor2Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color3Export
	void handleProcessExportEngineColor3Output();
	void handleProcessExportEngineColor3Error();
	void handleProcessExportEngineColor3Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	//Color4Export
	void handleProcessExportEngineColor4Output();
	void handleProcessExportEngineColor4Error();
	void handleProcessExportEngineColor4Finished(int exitCode, QProcess::ExitStatus exitStatus);
public slots:
	void cancelTrain();
};
