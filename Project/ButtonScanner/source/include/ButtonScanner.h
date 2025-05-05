#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ButtonScanner.h"

#include"DlgProduceLineSet.h"
#include"DlgProductSet.h"
#include"DlgExposureTimeSet.h"
#include"DlgNewProduction.h"
#include"DlgModelManager.h"
#include"rqw_LabelClickable.h"
#include"rqw_LabelWarning.h"
#include"PicturesViewer.h"

#include"opencv2/opencv.hpp"
#include<QImage>
#include<memory>

namespace rw
{
	namespace rqw
	{
		class CameraPassiveThread;
	}
}

QT_BEGIN_NAMESPACE
namespace Ui { class ButtonScannerClass; };
QT_END_NAMESPACE

class ButtonScanner : public QMainWindow
{
	Q_OBJECT
private:
	QLabel* label_lightBulb;
private:
	DlgProduceLineSet* _dlgProduceLineSet = nullptr;
	DlgProductSet* _dlgProductSet = nullptr;
	DlgExposureTimeSet* _dlgExposureTimeSet = nullptr;
	PicturesViewer* _picturesViewer = nullptr;
	DlgModelManager* _dlgModelManager = nullptr;
public:
	DlgNewProduction* dlgNewProduction = nullptr;
public:
	rw::rqw::ClickableLabel* labelClickable_title;
	rw::rqw::LabelWarning* labelWarning;
private:
	//变量监控线程关机的时候停止
	bool _mark_thread = false;
public:
	QRect exposureTimeTriggerArea; // 指定区域
	float exposureTimeTriggerWidthRatio = 0.3f;
	float exposureTimeTriggerRatio = 0.3f;
private:
	void updateExposureTimeTrigger();
	void onExposureTimeTriggerAreaClicked();
protected:
	void mousePressEvent(QMouseEvent* event)override;
	void resizeEvent(QResizeEvent* event) override;
public:
	ButtonScanner(QWidget* parent = nullptr);

	~ButtonScanner() override;

private:
	void set_radioButton();

private:
	void initializeComponents();
	void destroyComponents();

	void build_ui();
	void read_image();
	void build_mainWindowData();
	void build_dlgProduceLineSet();
	void build_dlgProductSet();
	void build_dlgExposureTimeSet();
	void build_dlgNewProduction();
	void build_modelStorageManager();
	void destroy_modelStorageManager();
	void build_picturesViewer();
	void build_dlgModelManager();

	void stop_all_axis();

	void build_connect();
private:
	//read_config必须在最前面运行
	void read_config();
	void read_config_mainWindowConfig();
	void read_config_produceLineConfig();
	void read_config_productSetConfig();
	void read_config_exposureTimeSetConfig();
	void read_config_hideScoreSet();

public:
	void build_imageSaveEngine();

	void clear_olderSavedImage();

	void build_camera();

	void build_imageProcessorModule();

	void start_monitor();

	void build_motion();

	//开启线程实施监控皮带运动位置
	void build_locationThread();
	//开启线程监控运动控制卡io点并且做出相应的逻辑
	void build_ioThread();

	void build_detachThread();
public:
	void showEvent(QShowEvent* event) override;
	

private:
	Ui::ButtonScannerClass* ui;

private:
	QImage cvMatToQImage(const cv::Mat& mat);
private:
	void onUpdateLightStateUi(size_t index, bool state);

private slots:
	void onCamera1Display(QPixmap image);

	void onCamera2Display(QPixmap image);

	void onCamera3Display(QPixmap image);

	void onCamera4Display(QPixmap image);
private slots:
	void updateStatisticalInfoUI();

private slots:
	void updateCameraLabelState(int cameraIndex, bool state);
	void updateCardLabelState(bool state);

private slots:
	void pbtn_exit_clicked();
	void pbtn_set_clicked();
	void pbtn_newProduction_clicked();
	void pbtn_beltSpeed_clicked();
	void pbtn_score_clicked();
	void pbtn_resetProduct_clicked();
	void pbtn_openSaveLocation_clicked();
private slots:
	void rbtn_debug_checked(bool checked);
	void rbtn_takePicture_checked(bool checked);
	void rbtn_removeFunc_checked(bool checked);
	void rbtn_upLight_checked(bool checked);
	void rbtn_sideLight_checked(bool checked);
	void rbtn_downLight_checked(bool checked);
	void rbtn_defect_checked(bool checked);
	void rbtn_forAndAgainst_checked(bool checked);
private:
	void labelClickable_title_clicked();
public slots:
	void onAddWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);
};
