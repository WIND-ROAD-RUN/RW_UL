#pragma once

#include <QDialog>
#include "ui_DlgProduceLineSet.h"
#include"MonitorIOState.hpp"
#include"DlgWarningManager.h"
#include"DlgWarningIOSetConfig.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProduceLineSetClass; };
QT_END_NAMESPACE

class DlgProduceLineSet : public QDialog
{
	Q_OBJECT
private:
	MonitorIOStateThread* monitorIoStateThread{nullptr};
	DlgWarningManager* dlgWarningManager{ nullptr };
	DlgWarningIOSetConfig* dlgWarningIOSetConfig{nullptr};
public:
	bool isDebug{false};
public:
	DlgProduceLineSet(QWidget* parent = nullptr);
	~DlgProduceLineSet();
protected:
	void showEvent(QShowEvent*) override;
private:
	void build_ui();
	void read_config();

	void build_connect();

	float get_blowTime();
public:
	void updateBeltSpeed();

private:
	Ui::DlgProduceLineSetClass* ui;

private slots:
	void pbtn_blowDistance1_clicked();
	void pbtn_blowDistance2_clicked();
	void pbtn_blowDistance3_clicked();
	void pbtn_blowDistance4_clicked();
	void pbtn_blowTime1_clicked();
	void pbtn_blowTime2_clicked();
	void pbtn_blowTime3_clicked();
	void pbtn_blowTime4_clicked();
	void pbtn_pixelEquivalent1_clicked();
	void pbtn_pixelEquivalent2_clicked();
	void pbtn_pixelEquivalent3_clicked();
	void pbtn_pixelEquivalent4_clicked();
	void pbtn_limit1_clicked();
	void pbtn_limit2_clicked();
	void pbtn_limit3_clicked();
	void pbtn_limit4_clicked();
	void pbtn_minBrightness_clicked();
	void pbtn_maxBrightness_clicked();
	void pbtn_motorSpeed_clicked();
	void pbtn_beltReductionRatio_clicked();
	void pbtn_accelerationAndDeceleration_clicked();
	void pbtn_codeWheel_clicked();
	void pbtn_pulseFactor_clicked();
	void pbtn_close_clicked();

	void cbox_workstationProtection12_checked(bool ischeck);
	void cbox_workstationProtection34_checked(bool ischeck);
	void cbox_debugMode_checked(bool ischeck);

	void cBox_takeMaskPictures(bool ischeck);
	void cBox_takeNgPictures(bool ischeck);
	void cBox_takeOkPictures(bool ischeck);

	void rbtn_drawCircle_clicked();
	void rbtn_drawRectangle_clicked();
private slots:
	void cbox_motoPower_checked(bool ischeck);
	void cbox_blow1_checked(bool ischeck);
	void cbox_blow2_checked(bool ischeck);
	void cbox_blow3_checked(bool ischeck);
	void cbox_blow4_checked(bool ischeck);
	void cbox_greenLight_checked(bool ischeck);
	void cbox_redLight_checked(bool ischeck);
	void cbox_upLight_checked(bool ischeck);
	void cbox_sideLight_checked(bool ischeck);
	void cbox_downLight_checked(bool ischeck);
	void cbox_storeLight_checked(bool ischeck);
	void cbox_beltControl(bool ischeck);
private slots:
	void cbox_DIStart_checked(bool ischeck);
	void cbox_DIStop_checked(bool ischeck);
	void cbox_DIShutdownComputer_checked(bool ischeck);
	void cbox_DIAirPressure_checked(bool ischeck);
	void cbox_DICameraTrigger1_checked(bool ischeck);
	void cbox_DICameraTrigger2_checked(bool ischeck);
	void cbox_DICameraTrigger3_checked(bool ischeck);
	void cbox_DICameraTrigger4_checked(bool ischeck);
private slots:
	void onDIState(int index, bool state);
	void onDOState(int index, bool state);
private slots:
	void pbtn_warningManager_clicked();
	void pbtn_DIOValueSet_clicked();
};
