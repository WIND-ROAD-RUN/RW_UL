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
	void build_connect();
	float get_blowTime();
	void read_image();
	void build_radioButton();
public:
	void readConfig();

private:
	Ui::DlgProductSetClass* ui;
private slots:
	void pbtn_close_clicked();
	void pbtn_blowTime_clicked();
	void pbtn_photography_clicked();

private slots:
	void clickedLabel_clicked();

private slots:
	//外径
	void rbtn_outsideDiameterEnable_checked(bool checked);
	void pbtn_outsideDiameterValue_clicked();
	void pbtn_outsideDiameterDeviation_clicked();

	//屏蔽范围
	void rbtn_shieldingRangeEnable_checked(bool checked);
	void pbtn_outerRadius_clicked();
	void pbtn_innerRadius_clicked();

	//孔数
	void rbtn_holesCountEnable_checked(bool checked);
	void ptn_holesCountValue_clicked();

	//破眼
	void rbtn_brokenEyeEnable_checked(bool checked);
	void pbtn_brokenEyeSimilarity_clicked();

	//裂痕
	void rbtn_crackEnable_checked(bool checked);
	void pbtn_crackSimilarity_clicked();

	//孔径
	void rbtn_apertureEnable_checked(bool checked);
	void pbtn_apertureValue_clicked();
	void pbtn_apertureSimilarity_clicked();

	//孔心距
	void rbtn_holeCenterDistanceEnable_checked(bool checked);
	void pbtn_holeCenterDistanceValue_clicked();
	void pbtn_holeCenterDistanceSimilarity_clicked();

	//指定色差
	void rbtn_specifyColorDifferenceEnable_checked(bool checked);
	void pbtn_specifyColorDifferenceR_clicked();
	void pbtn_specifyColorDifferenceG_clicked();
	void pbtn_specifyColorDifferenceB_clicked();
	void pbtn_specifyColorDifferenceDeviation_clicked();

	//大色差
	void rbtn_largeColorDifferenceEnable_checked(bool checked);
	void pbtn_largeColorDifferenceDeviation_clicked();

	//破边
	void rbtn_edgeDamageEnable_checked(bool checked);
	void pbtn_edgeDamageSimilarity_clicked();
	void pbtn_edgeDamageArea_clicked();

	//崩口
	void rbtn_bengKou_checked(bool checked);
	void pbtn_bengKouScore_clicked();

	//气孔
	void rbtn_poreEnable_checked(bool checked);
	void pbtn_poreEnableScore_clicked();
	void pbtn_poreEnableArea_clicked();

	//小气孔
	void rbtn_smallPoreEnable_checked(bool checked);
	void pbtn_smallPoreEnableScore_clicked();
	void pbtn_smallPoreEnableArea_clicked();

	//油漆
	void rbtn_paintEnable_checked(bool checked);
	void pbtn_paintEnableScore_clicked();

	//魔石
	void rbtn_grindStoneEnable_checked(bool checked);
	void pbtn_grindStoneScore_clicked();

	//堵眼
	void rbtn_blockEyeEnable_checked(bool checked);
	void pbtn_blockEyeScore_clicked();

	//料头
	void rbtn_materialHeadEnable_checked(bool checked);
	void pbtn_materialHeadScore_clicked();
};
