#pragma once

#include <QDialog>
#include "ui_DlgNewProduction.h"
#include"PicturesViewer.h"

#include"opencv2/opencv.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgNewProductionClass; };
QT_END_NAMESPACE

struct DlgNewProductionInfo
{
	enum State
	{
		None,
		CheckBladeShape,
		CheckColor
	}state;

	std::atomic_bool isActivate{ false };
	size_t currentTabIndex{ 0 };
};

class DlgNewProduction : public QDialog
{
	Q_OBJECT
public:
	DlgNewProduction(QWidget* parent = nullptr);

	~DlgNewProduction();
public:
	PicturesViewer* picturesViewer;
private:
	void build_ui();
	void build_connect();

private:
	Ui::DlgNewProductionClass* ui;

private:
	void set_motionRun(bool isRun);

private:
	void build_dialog();
	void destroy();

public:
	DlgNewProductionInfo _info;
private:
	bool _trainSate{ false };
public slots:
	void updateTrainState(bool isTrain);
public slots:
	void appendAiTrainLog(QString log);
	void updateProgress(int value, int total);
	void updateProgressTitle(QString s);
public slots:
	void img_display_work(cv::Mat frame, size_t index);
private:
	void img_display_work1(const QPixmap& pixmap);
	void img_display_work2(const QPixmap& pixmap);
	void img_display_work3(const QPixmap& pixmap);
	void img_display_work4(const QPixmap& pixmap);
private slots:
	void pbtn_tab1_ok_clicked();
	void pbtn_tab1_no_clicked();
	void pbtn_tab1_exit_clicked();
private slots:
	void pbtn_tab2_check_color_clicked();
	void pbtn_tab2_check_blade_shape_clicked();
	void pbtn_tab2_pre_step_clicked();
	void pbtn_tab2_exit_clicked();
private slots:
	void pbtn_tab3_open_img_locate_clicked();
	void pbtn_tab3_exit_clicked();
	void pbtn_tab3_pre_step_clicked();
	void pbtn_tab3_nex_step_clicked();
private slots:
	void pbtn_tab4_open_img_locate_clicked();
	void pbtn_tab4_exit_clicked();
	void pbtn_tab4_pre_step_clicked();
	void pbtn_tab4_nex_step_clicked();
private slots:
	void pbtn_tab5_start_train_clicked();
	void pbtn_tab5_exit_clicked();
	void pbtn_tab5_pre_step_clicked();
	void pbtn_tab5_finish_clicked();
	void pbtn_tab5_cancelTrain_clicked();
signals:
	void cancelTrain();
protected:
	void showEvent(QShowEvent*) override;
public slots:
	void flashImgCount();
};