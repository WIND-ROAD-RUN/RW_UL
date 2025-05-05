#pragma once

#include <QDialog>
#include "ui_DlgProductSet.h"
#include"rqw_LabelClickable.h"
#include"DlgHideScoreSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductSetClass; };
QT_END_NAMESPACE

class DlgProductSet : public QDialog
{
	Q_OBJECT
private:
	DlgHideScoreSet* _dlgHideScoreSet;
public:
	DlgProductSet(QWidget* parent = nullptr);
	~DlgProductSet();
private:
	rw::rqw::ClickableLabel* _clickedLabel;
private:
	void build_ui();
public:
	void readConfig();
private:
	float get_blowTime();
	void read_image();
	void build_radioButton();
	void build_connect();
private:
	Ui::DlgProductSetClass* ui;
private slots:
	void pbtn_close_clicked();

	void pbtn_outsideDiameterValue_clicked();
	void pbtn_outsideDiameterDeviation_clicked();
	void pbtn_photography_clicked();
	void pbtn_blowTime_clicked();
	void pbtn_outerRadius_clicked();
	void pbtn_innerRadius_clicked();
	void ptn_holesCountValue_clicked();
	void pbtn_brokenEyeSimilarity_clicked();
	void pbtn_crackSimilarity_clicked();
	void pbtn_apertureValue_clicked();
	void pbtn_apertureSimilarity_clicked();
	void pbtn_holeCenterDistanceValue_clicked();
	void pbtn_holeCenterDistanceSimilarity_clicked();
	void pbtn_specifyColorDifferenceR_clicked();
	void pbtn_specifyColorDifferenceG_clicked();
	void pbtn_specifyColorDifferenceB_clicked();
	void pbtn_specifyColorDifferenceDeviation_clicked();
	void pbtn_largeColorDifferenceDeviation_clicked();
	void pbtn_edgeDamageSimilarity_clicked();
private slots:
	void pbtn_poreEnableScore_clicked();
	void pbtn_paintEnableScore_clicked();
	void pbtn_grindStoneScore_clicked();
	void pbtn_blockEyeScore_clicked();
	void pbtn_materialHeadScore_clicked();

private slots:
	void rbtn_outsideDiameterEnable_checked(bool checked);
	void rbtn_edgeDamageEnable_checked(bool checked);
	void rbtn_shieldingRangeEnable_checked(bool checked);
	void rbtn_poreEnable_checked(bool checked);
	void rbtn_paintEnable_checked(bool checked);
	void rbtn_holesCountEnable_checked(bool checked);
	void rbtn_brokenEyeEnable_checked(bool checked);
	void rbtn_crackEnable_checked(bool checked);
	void rbtn_apertureEnable_checked(bool checked);
	void rbtn_holeCenterDistanceEnable_checked(bool checked);
	void rbtn_specifyColorDifferenceEnable_checked(bool checked);
	void rbtn_largeColorDifferenceEnable_checked(bool checked);
	void rbtn_grindStoneEnable_checked(bool checked);
	void rbtn_blockEyeEnable_checked(bool checked);
	void rbtn_materialHeadEnable_checked(bool checked);
private slots:
	void clickedLabel_clicked();
};