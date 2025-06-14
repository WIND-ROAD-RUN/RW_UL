#pragma once

#include <QMainWindow>

#include "DlgProductScore.h"
#include "DlgProductSet.h"
#include "ui_SmartCroppingOfBags.h"
#include "PictureViewerThumbnails.h"
#include"rqw_CarouselWidget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class SmartCroppingOfBagsClass; };
QT_END_NAMESPACE

class SmartCroppingOfBags : public QMainWindow
{
	Q_OBJECT
private:
	CarouselWidget* _carouselWidget = nullptr;
public:
	SmartCroppingOfBags(QWidget *parent = nullptr);
	~SmartCroppingOfBags();

public:
	DlgProductSetSmartCroppingOfBags* _dlgProductSet = nullptr;
	DlgProductScoreSmartCroppingOfBags* _dlgProductScore = nullptr;
	//DlgExposureTimeSet* _dlgExposureTimeSet = nullptr;

private:
	PictureViewerThumbnails* _picturesViewer = nullptr;

public:
	void build_ui();
	void build_connect();
	void build_motion();
	void destroy_motion();
	void build_camera();

	void build_SmartCroppingOfBagsData();
	void build_DlgProductSetData();
	void build_DlgProductScore();

	void build_imageProcessorModule();
	void build_imageSaveEngine();

public:
	void destroyComponents();

public:
	void read_config();
	void read_config_GeneralConfig();
	void read_config_ScoreConfig();
	void read_config_SetConfig();

private slots:
	void btn_close_clicked();

	void btn_pingbiquyu_clicked();
	void btn_chanliangqingling_clicked();
	void btn_daizizhonglei_clicked();
	void btn_down_clicked();
	void btn_up_clicked();
	void btn_baoguang_clicked();
	void btn_normalParam_clicked();
	void btn_setParam_clicked();

	void ckb_tifei_checked();
	void ckb_Debug_checked(bool checked);
	void ckb_cuntu_checked();
	void rbtn_zhinengcaiqie_clicked(bool checked);
	void rbtn_yinshuazhiliangjiance_clicked(bool checked);

private slots:
	void updateCameraLabelState(int cameraIndex, bool state);

	void onCamera1Display(QPixmap image);
	void onCamera2Display(QPixmap image);

	void onCameraNGDisplay(QPixmap image, size_t index, bool isbad);

	void updateCardLabelState(bool state);

private:
	Ui::SmartCroppingOfBagsClass *ui;
};

