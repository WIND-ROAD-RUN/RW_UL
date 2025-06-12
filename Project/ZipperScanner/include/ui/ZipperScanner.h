#pragma once

#include <QMainWindow>
#include "ui_ZipperScanner.h"
#include "DlgProductSet.h"
#include "DlgProductScore.h"
#include "DlgExposureTimeSet.h"
#include <rqw_LabelWarning.h>
#include <opencv2/core/mat.hpp>

#include "PictureViewerThumbnails.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ZipperScannerClass; };
QT_END_NAMESPACE

class ZipperScanner : public QMainWindow
{
	Q_OBJECT

public:
	ZipperScanner(QWidget *parent = nullptr);
	~ZipperScanner();

public:
	DlgProductSet* _dlgProductSet = nullptr;
	DlgProductScore* _dlgProductScore = nullptr;
	DlgExposureTimeSet* _dlgExposureTimeSet = nullptr;

private:
	PictureViewerThumbnails* _picturesViewer = nullptr;

public :
	void build_ui();
	void build_connect();
	void build_camera();
    
	void build_ZipperScannerData();
	void build_DlgProductSetData();
	void build_DlgProductScore();
	void build_DlgExposureTimeSet();

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
	void pbtn_exit_clicked();
	void pbtn_set_clicked();
	void pbtn_score_clicked();
	void rbtn_debug_checked(bool checked);
	void rbtn_strongLight_checked(bool checked);
	void rbtn_mediumLight_checked(bool checked);
	void rbtn_weakLight_checked(bool checked);
	void pbtn_openSaveLocation_clicked();
	void rbtn_takePicture_checked();
	void rbtn_removeFunc_checked(bool checked);


private slots:
	void updateCameraLabelState(int cameraIndex, bool state);

	void onCamera1Display(QPixmap image);

	void onCamera2Display(QPixmap image);

private:
	Ui::ZipperScannerClass *ui;
};
