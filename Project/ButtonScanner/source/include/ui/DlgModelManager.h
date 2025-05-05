#pragma once

#include <QDialog>

#include "cdm_AiModelConfig.h"
#include"cdm_AiModelConfigIndex.h"
#include "ui_DlgModelManager.h"
#include"LoadingDialog.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgModelManagerClass; };
QT_END_NAMESPACE

class DlgModelManager : public QDialog
{
	Q_OBJECT

public:
	DlgModelManager(QWidget* parent = nullptr);
	~DlgModelManager();
private:
	QStringListModel* _ModelListModel;
	QStandardItemModel* _ModelInfoModel;
	LoadingDialog* _loadingDialog;
	rw::cdm::AiModelConfigIndex _configIndex;

private:
	void build_ui();
	void build_connect();

private:
	Ui::DlgModelManagerClass* ui;

protected:
	void showEvent(QShowEvent*) override;
private:
	QVector<rw::cdm::AiModelConfig> _modelConfigs;
private:
	QString formatDateString(const std::string& dateStr);
	QVector<QString> getImagePaths(const QString& rootPath, bool isGood);
	QVector<QString> getImagePaths(const QString& rootPath, bool isGood, int maxCount);
	void deleteDirectory(const QString& targetPath);
	bool copyDirectoryRecursively(const QString& sourceDirPath, const QString& targetDirPath);

private:
	QString findXmlFile(const QString& rootPath);

private:
	void flashModelList();
	void flashModelInfoTable(size_t index);
	void flashExampleImage(size_t index);

private:
	void copyTargetImageFromStorageInTemp();
	void copyOOModel();
	void copySOModel();

private slots:
	void pbtn_exit_clicked();
	void onModelListSelectionChanged(const QModelIndex& current, const QModelIndex& previous);
	void pbtn_nextModel_clicked();
	void pbtn_preModel_clicked();
	void pbtn_deleteModel_clicked();
	void pbtn_loadModel_clicked();
};
