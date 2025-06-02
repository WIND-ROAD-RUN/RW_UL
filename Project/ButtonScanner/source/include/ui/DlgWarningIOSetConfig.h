#pragma once

#include <QDialog>
#include "ui_DlgWarningIOSetConfig.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgWarningIOSetConfigClass; };
QT_END_NAMESPACE

class DlgWarningIOSetConfig : public QDialog
{
	Q_OBJECT
private:
	std::vector<std::vector<int>> DIFindAllDuplicateIndices();
	void setDIErrorInfo(const std::vector<std::vector<int>>& index);
	void setDIErrorInfo(int index);
private:
	std::vector<std::vector<int>> DOFindAllDuplicateIndices();
	void setDOErrorInfo(const std::vector<std::vector<int>>& index);
	void setDOErrorInfo(int index);
public:
	DlgWarningIOSetConfig(QWidget *parent = nullptr);
	~DlgWarningIOSetConfig();
private:
	void build_ui();
	void build_connect();
	void read_config();

private:
	Ui::DlgWarningIOSetConfigClass *ui;
public slots:
	void pbtn_exit_clicked();
public slots:
	//DI
	void pbtn_DIStartValue_clicked();
	void pbtn_DIStopValue_clicked();
	void pbtn_DIShutdownComputerValue_clicked();
	void pbtn_DIAirPressureValue_clicked();
	void pbtn_DICameraTriggerValue1_clicked();
	void pbtn_DICameraTriggerValue2_clicked();
	void pbtn_DICameraTriggerValue3_clicked();
	void pbtn_DICameraTriggerValue4_clicked();
public slots:
	//DO
	void pbtn_DOMotoPowerValue_clicked();
	void pbtn_DOBlowTime1Value_clicked();
	void pbtn_DOBlowTime2Value_clicked();
	void pbtn_DOBlowTime3Value_clicked();
	void pbtn_DOBlowTime4Value_clicked();
	void pbtn_DOGreenLightValue_clicked();
	void pbtn_DORedLightValue_clicked();
	void pbtn_DOUpLightValue_clicked();
	void pbtn_DOSideLightValue_clicked();
	void pbtn_DODownLightValue_clicked();
	void pbtn_DOStoreLightValue_clicked();
	void pbtn_DOStartBelt_clicked();
};
