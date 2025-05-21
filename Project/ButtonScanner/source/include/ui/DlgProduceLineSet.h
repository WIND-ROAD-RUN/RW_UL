#pragma once

#include <QDialog>
#include "ui_DlgProduceLineSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProduceLineSetClass; };
QT_END_NAMESPACE

class DlgProduceLineSet : public QDialog
{
	Q_OBJECT
public:
	bool isDebug{false};
public:
	DlgProduceLineSet(QWidget* parent = nullptr);
	~DlgProduceLineSet();

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
	void cbox_DO0_checked(bool ischeck);
	void cbox_DO1_checked(bool ischeck);
	void cbox_DO2_checked(bool ischeck);
	void cbox_DO3_checked(bool ischeck);
	void cbox_DO4_checked(bool ischeck);
	void cbox_DO5_checked(bool ischeck);
	void cbox_DO6_checked(bool ischeck);
	void cbox_DO7_checked(bool ischeck);
	void cbox_DO8_checked(bool ischeck);
	void cbox_DO9_checked(bool ischeck);
	void cbox_DO10_checked(bool ischeck);
	void cbox_beltControl(bool ischeck);
private slots:
	void cbox_DI0_checked(bool ischeck);
	void cbox_DI1_checked(bool ischeck);
	void cbox_DI2_checked(bool ischeck);
	void cbox_DI3_checked(bool ischeck);
	void cbox_DI4_checked(bool ischeck);
	void cbox_DI5_checked(bool ischeck);
	void cbox_DI6_checked(bool ischeck);
	void cbox_DI7_checked(bool ischeck);
	void cbox_DI8_checked(bool ischeck);
	void cbox_DI9_checked(bool ischeck);
	void cbox_DI10_checked(bool ischeck);
};
