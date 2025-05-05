#include "stdafx.h"
#include "DlgNewProduction.h"

#include"ButtonUtilty.h"
#include"GlobalStruct.h"
#include"PicturesViewer.h"
#include<QThread>
#include<QtConcurrent>

#include"imeot_ModelEngineOT.h"
#include"imest_ModelEnginest.h"
#include"rqw_CameraObjectThread.hpp"
#include"rqw_CameraObject.hpp"

DlgNewProduction::DlgNewProduction(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgNewProductionClass())
{
	ui->setupUi(this);
	build_ui();
	build_connect();
}

DlgNewProduction::~DlgNewProduction()
{
	delete ui;
}

void DlgNewProduction::build_ui()
{
	ui->tabWidget->tabBar()->hide();
	build_dialog();
	picturesViewer = new PicturesViewer(this);
	auto tempImagePath = globalPath.modelStorageManagerTempPath + R"(Image\)";
	picturesViewer->setRootPath(tempImagePath);
}

void DlgNewProduction::build_connect()
{
	//tab1
	QObject::connect(ui->pbtn_tab1_exit, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab1_exit_clicked);
	QObject::connect(ui->pbtn_tab1_no, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab1_no_clicked);
	QObject::connect(ui->pbtn_tab1_ok, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab1_ok_clicked);
	//tab2
	QObject::connect(ui->pbtn_tab2_checkColor, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab2_check_color_clicked);
	QObject::connect(ui->pbtn_tab2_checkBladeShape, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab2_check_blade_shape_clicked);
	QObject::connect(ui->pbtn_tab2_preStep, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab2_pre_step_clicked);
	QObject::connect(ui->pbtn_tab2_exit, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab2_exit_clicked);
	//tab3
	QObject::connect(ui->pbtn_tab3_openImgLocate, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab3_open_img_locate_clicked);
	QObject::connect(ui->pbtn_tab3_exit, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab3_exit_clicked);
	QObject::connect(ui->pbtn_tab3_preStep, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab3_pre_step_clicked);
	QObject::connect(ui->pbtn_tab3_nexStep, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab3_nex_step_clicked);
	//tab4
	QObject::connect(ui->pbtn_tab4_openImgLocate, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab4_open_img_locate_clicked);
	QObject::connect(ui->pbtn_tab4_exit, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab4_exit_clicked);
	QObject::connect(ui->pbtn_tab4_preStep, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab4_pre_step_clicked);
	QObject::connect(ui->pbtn_tab4_nexStep, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab4_nex_step_clicked);
	//tab5
	QObject::connect(ui->pbtn_tab5_startTrain, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab5_start_train_clicked);
	QObject::connect(ui->pbtn_tab5_exit, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab5_exit_clicked);
	QObject::connect(ui->pbtn_tab5_preStep, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab5_pre_step_clicked);
	QObject::connect(ui->pbtn_tab5_finish, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab5_finish_clicked);
	QObject::connect(ui->pbtn_tab5_cancelTrain, &QPushButton::clicked,
		this, &DlgNewProduction::pbtn_tab5_cancelTrain_clicked);

	QObject::connect(picturesViewer, &PicturesViewer::viewerClosed,
		this, &DlgNewProduction::flashImgCount);
}

void DlgNewProduction::set_motionRun(bool isRun)
{
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	if (isRun)
	{
		motionPtr->SetIOOut(1, true);
		motionPtr->SetAxisType(0, 1);
		double unit = GlobalStructData::getInstance().dlgProduceLineSetConfig.pulseFactor;
		motionPtr->SetAxisPulse(0, unit);
		double acc = GlobalStructData::getInstance().dlgProduceLineSetConfig.accelerationAndDeceleration;
		motionPtr->SetAxisAcc(0, acc);
		motionPtr->SetAxisDec(0, acc);
		double speed = GlobalStructData::getInstance().dlgProduceLineSetConfig.motorSpeed;
		motionPtr->SetAxisRunSpeed(0, speed);
		// pidaimove->start(100);
		motionPtr->AxisRun(0, -1);
		motionPtr->SetIOOut(7, true);
	}
	else
	{
		motionPtr->StopAllAxis();
		motionPtr->SetIOOut(1, isRun);
		motionPtr->SetIOOut(7, isRun);
	}
}

void DlgNewProduction::build_dialog()
{
	ui->tabWidget->setCurrentIndex(0);

	QButtonGroup* tab3ChoiceGroup = new QButtonGroup(this);

	tab3ChoiceGroup->addButton(ui->rbtn_tab3_checkBladeShape);
	tab3ChoiceGroup->addButton(ui->rbtn_tab3_filterColor);

	QButtonGroup* tab3WorkGroup = new QButtonGroup(this);
	tab3WorkGroup->addButton(ui->rbtn_tab3_firstWork1);
	tab3WorkGroup->addButton(ui->rbtn_tab3_firstWork2);
	tab3WorkGroup->addButton(ui->rbtn_tab3_firstWork3);
	tab3WorkGroup->addButton(ui->rbtn_tab3_firstWork4);

	QButtonGroup* tab4ChoiceGroup = new QButtonGroup(this);

	tab4ChoiceGroup->addButton(ui->rbtn_tab4_checkBladeShape);
	tab4ChoiceGroup->addButton(ui->rbtn_tab4_filterColor);

	QButtonGroup* tab4WorkGroup = new QButtonGroup(this);
	tab4WorkGroup->addButton(ui->rbtn_tab4_firstWork1);
	tab4WorkGroup->addButton(ui->rbtn_tab4_firstWork2);
	tab4WorkGroup->addButton(ui->rbtn_tab4_firstWork3);
	tab4WorkGroup->addButton(ui->rbtn_tab4_firstWork4);
}

void DlgNewProduction::destroy()
{
	set_motionRun(false);
	_info.state = DlgNewProductionInfo::None;
	ui->tabWidget->setCurrentIndex(0);
	this->_info.currentTabIndex = 0;
	this->_info.isActivate = false;
	this->hide();
}

void DlgNewProduction::img_display_work1(const QPixmap& pixmap)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;
	if (pixmap.isNull())
	{
		return;
	}
	if (this->_info.currentTabIndex == 2 && ui->rbtn_tab3_firstWork1->isChecked())
	{
		modelStorageManager->work1_bad_count_ += 1;
		modelStorageManager->save_work1_image(pixmap.toImage(), false);
		ui->label_tab3_tabImgCount1->setText(QString::number(modelStorageManager->work1_bad_count_));
		ui->label_tab3_imgDisplay1->setPixmap(pixmap.scaled(ui->label_tab3_imgDisplay1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else if (this->_info.currentTabIndex == 3 && ui->rbtn_tab4_firstWork1->isChecked())
	{
		modelStorageManager->work1_good_count_ += 1;
		modelStorageManager->save_work1_image(pixmap.toImage(), true);
		ui->label_tab4_tabImgCount1->setText(QString::number(modelStorageManager->work1_good_count_));
		ui->label_tab4_imgDisplay1->setPixmap(pixmap.scaled(ui->label_tab4_imgDisplay1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
}

void DlgNewProduction::img_display_work2(const QPixmap& pixmap)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;
	if (pixmap.isNull())
	{
		return;
	}
	if (this->_info.currentTabIndex == 2 && ui->rbtn_tab3_firstWork2->isChecked())
	{
		modelStorageManager->work2_bad_count_ += 1;
		modelStorageManager->save_work2_image(pixmap.toImage(), false);
		ui->label_tab3_tabImgCount2->setText(QString::number(modelStorageManager->work2_bad_count_));
		ui->label_tab3_imgDisplay2->setPixmap(pixmap.scaled(ui->label_tab3_imgDisplay2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else if (this->_info.currentTabIndex == 3 && ui->rbtn_tab4_firstWork2->isChecked())
	{
		modelStorageManager->work2_good_count_ += 1;
		modelStorageManager->save_work2_image(pixmap.toImage(), true);
		ui->label_tab4_tabImgCount2->setText(QString::number(modelStorageManager->work2_good_count_));
		ui->label_tab4_imgDisplay2->setPixmap(pixmap.scaled(ui->label_tab4_imgDisplay2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
}

void DlgNewProduction::img_display_work3(const QPixmap& pixmap)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;
	if (pixmap.isNull())
	{
		return;
	}
	if (this->_info.currentTabIndex == 2 && ui->rbtn_tab3_firstWork3->isChecked())
	{
		modelStorageManager->work3_bad_count_ += 1;
		modelStorageManager->save_work3_image(pixmap.toImage(), false);
		ui->label_tab3_tabImgCount3->setText(QString::number(modelStorageManager->work3_bad_count_));
		ui->label_tab3_imgDisplay3->setPixmap(pixmap.scaled(ui->label_tab3_imgDisplay3->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else if (this->_info.currentTabIndex == 3 && ui->rbtn_tab4_firstWork3->isChecked())
	{
		modelStorageManager->work3_good_count_ += 1;
		modelStorageManager->save_work3_image(pixmap.toImage(), true);
		ui->label_tab4_tabImgCount3->setText(QString::number(modelStorageManager->work3_good_count_));
		ui->label_tab4_imgDisplay3->setPixmap(pixmap.scaled(ui->label_tab4_imgDisplay3->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
}

void DlgNewProduction::img_display_work4(const QPixmap& pixmap)
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;
	if (pixmap.isNull())
	{
		return;
	}
	if (this->_info.currentTabIndex == 2 && ui->rbtn_tab3_firstWork4->isChecked())
	{
		modelStorageManager->work4_bad_count_ += 1;
		modelStorageManager->save_work4_image(pixmap.toImage(), false);
		ui->label_tab3_tabImgCount4->setText(QString::number(modelStorageManager->work4_bad_count_));
		ui->label_tab3_imgDisplay4->setPixmap(pixmap.scaled(ui->label_tab3_imgDisplay4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else if (this->_info.currentTabIndex == 3 && ui->rbtn_tab4_firstWork4->isChecked())
	{
		modelStorageManager->work4_good_count_ += 1;
		modelStorageManager->save_work4_image(pixmap.toImage(), true);
		ui->label_tab4_tabImgCount4->setText(QString::number(modelStorageManager->work4_good_count_));\
			ui->label_tab4_imgDisplay4->setPixmap(pixmap.scaled(ui->label_tab4_imgDisplay4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
}

void DlgNewProduction::pbtn_tab1_no_clicked()
{
	ui->tabWidget->setCurrentIndex(1);
	this->_info.currentTabIndex = 1;
}

void DlgNewProduction::pbtn_tab1_exit_clicked()
{
	destroy();
}

void DlgNewProduction::pbtn_tab2_check_color_clicked()
{
	ui->rbtn_tab3_filterColor->setChecked(true);
	ui->rbtn_tab4_filterColor->setChecked(true);
	ui->rbtn_tab5_filterColor->setChecked(true);
	this->_info.state = _info.state = DlgNewProductionInfo::CheckColor;
	ui->tabWidget->setCurrentIndex(2);
	this->_info.currentTabIndex = 2;

	auto& modelStorageManager = GlobalStructData::getInstance().modelStorageManager;
	ui->label_tab3_tabImgCount1->setText(QString::number(modelStorageManager->work1_bad_count_));
	ui->label_tab3_tabImgCount2->setText(QString::number(modelStorageManager->work2_bad_count_));
	ui->label_tab3_tabImgCount3->setText(QString::number(modelStorageManager->work3_bad_count_));
	ui->label_tab3_tabImgCount4->setText(QString::number(modelStorageManager->work4_bad_count_));
	set_motionRun(true);
}

void DlgNewProduction::pbtn_tab2_check_blade_shape_clicked()
{
	ui->rbtn_tab3_filterColor->setChecked(false);
	ui->rbtn_tab4_filterColor->setChecked(false);
	ui->rbtn_tab5_filterColor->setChecked(false);
	ui->rbtn_tab3_checkBladeShape->setChecked(true);
	ui->rbtn_tab4_checkBladeShape->setChecked(true);
	ui->rbtn_tab5_checkBladeShape->setChecked(true);
	this->_info.state = _info.state = DlgNewProductionInfo::CheckBladeShape;
	ui->tabWidget->setCurrentIndex(2);
	this->_info.currentTabIndex = 2;

	auto& modelStorageManager = GlobalStructData::getInstance().modelStorageManager;
	ui->label_tab3_tabImgCount1->setText(QString::number(modelStorageManager->work1_bad_count_));
	ui->label_tab3_tabImgCount2->setText(QString::number(modelStorageManager->work2_bad_count_));
	ui->label_tab3_tabImgCount3->setText(QString::number(modelStorageManager->work3_bad_count_));
	ui->label_tab3_tabImgCount4->setText(QString::number(modelStorageManager->work4_bad_count_));
	set_motionRun(true);
}

void DlgNewProduction::pbtn_tab2_pre_step_clicked()
{
	ui->tabWidget->setCurrentIndex(0);
	this->_info.currentTabIndex = 0;
}

void DlgNewProduction::pbtn_tab2_exit_clicked()
{
	destroy();
}

void DlgNewProduction::pbtn_tab3_open_img_locate_clicked()
{
	picturesViewer->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	picturesViewer->show();
}

void DlgNewProduction::pbtn_tab3_exit_clicked()
{
	destroy();
}

void DlgNewProduction::pbtn_tab3_pre_step_clicked()
{
	ui->tabWidget->setCurrentIndex(1);
	this->_info.currentTabIndex = 1;
	this->_info.state = _info.state = DlgNewProductionInfo::None;
	ui->rbtn_tab3_filterColor->setChecked(false);
	ui->rbtn_tab4_filterColor->setChecked(false);
	ui->rbtn_tab5_filterColor->setChecked(false);
	ui->rbtn_tab3_checkBladeShape->setChecked(true);
	ui->rbtn_tab4_checkBladeShape->setChecked(true);
	ui->rbtn_tab5_checkBladeShape->setChecked(true);
	set_motionRun(false);
}

void DlgNewProduction::pbtn_tab3_nex_step_clicked()
{
	ui->tabWidget->setCurrentIndex(3);
	this->_info.currentTabIndex = 3;

	auto& modelStorageManager = GlobalStructData::getInstance().modelStorageManager;
	ui->label_tab4_tabImgCount1->setText(QString::number(modelStorageManager->work1_good_count_));
	ui->label_tab4_tabImgCount2->setText(QString::number(modelStorageManager->work2_good_count_));
	ui->label_tab4_tabImgCount3->setText(QString::number(modelStorageManager->work3_good_count_));
	ui->label_tab4_tabImgCount4->setText(QString::number(modelStorageManager->work4_good_count_));
}

void DlgNewProduction::pbtn_tab4_open_img_locate_clicked()
{
	picturesViewer->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	picturesViewer->show();
}

void DlgNewProduction::pbtn_tab4_exit_clicked()
{
	destroy();
}

void DlgNewProduction::pbtn_tab4_pre_step_clicked()
{
	ui->tabWidget->setCurrentIndex(2);
	this->_info.currentTabIndex = 2;
}

void DlgNewProduction::pbtn_tab4_nex_step_clicked()
{
	ui->tabWidget->setCurrentIndex(4);
	this->_info.currentTabIndex = 4;
	set_motionRun(false);
}

void DlgNewProduction::pbtn_tab5_start_train_clicked()
{
	auto& aiTrainModule = GlobalStructThread::getInstance().aiTrainModule;
	if (this->_info.state == DlgNewProductionInfo::CheckBladeShape)
	{
		aiTrainModule->setModelType(ModelType::ObjectDetection);
	}
	else if (this->_info.state == DlgNewProductionInfo::CheckColor)
	{
		aiTrainModule->setModelType(ModelType::Segment);
	}
	else
	{
		return;
	}
	aiTrainModule->startTrain();
}

void DlgNewProduction::pbtn_tab5_exit_clicked()
{
	destroy();
}

void DlgNewProduction::pbtn_tab5_pre_step_clicked()
{
	ui->tabWidget->setCurrentIndex(3);
	this->_info.currentTabIndex = 3;
	set_motionRun(true);
}

void DlgNewProduction::pbtn_tab5_finish_clicked()
{
	destroy();
}

void DlgNewProduction::pbtn_tab5_cancelTrain_clicked()
{
	auto result = QMessageBox::question(this, "确认", "你真的要终止训练吗，若终止的话当前的训练结果将会丢失");
	if (result == QMessageBox::Yes)
	{
		emit cancelTrain();
	}
}

void DlgNewProduction::showEvent(QShowEvent* show_event)
{
	this->_info.isActivate = true;
	QDialog::showEvent(show_event);

	if (_trainSate == true)
	{
		ui->tabWidget->setCurrentIndex(4);
		this->_info.currentTabIndex = 4;
		ui->pbtn_tab5_cancelTrain->setEnabled(true);
	}
	else
	{
		ui->label_trainState->setText("未开始训练");
		ui->progressBar_tab5->setValue(0);
		ui->plainTextEdit_tab5->clear();
		ui->pbtn_tab5_cancelTrain->setEnabled(false);
	}
}

void DlgNewProduction::updateTrainState(bool isTrain)
{
	_trainSate = isTrain;
	ui->pbtn_tab5_finish->setEnabled(!isTrain);
	ui->pbtn_tab5_startTrain->setEnabled(!isTrain);
	ui->pbtn_tab5_preStep->setEnabled(!isTrain);
	ui->pbtn_tab5_cancelTrain->setEnabled(isTrain);
}

void DlgNewProduction::appendAiTrainLog(QString log)
{
	if (log.isEmpty())
	{
		return;
	}
	ui->plainTextEdit_tab5->appendPlainText(log);
}

void DlgNewProduction::updateProgress(int value, int total)
{
	int progress = static_cast<double>(value) / static_cast<double>(total) * 100;
	ui->progressBar_tab5->setValue(progress);
}

void DlgNewProduction::updateProgressTitle(QString s)
{
	if (s.isEmpty())
	{
		return;
	}
	ui->label_trainState->setText(s);
}

void DlgNewProduction::img_display_work(cv::Mat frame, size_t index)
{
	if (_info.isActivate == false)
	{
		return;
	}
	if ((_info.currentTabIndex != 2) && (_info.currentTabIndex != 3))
	{
		return;
	}
	auto pixmap = cvMatToQPixmap(frame);
	switch (index)
	{
	case 1:
		img_display_work1(pixmap);
		break;
	case 2:
		img_display_work2(pixmap);
		break;
	case 3:
		img_display_work3(pixmap);
		break;
	case 4:
		img_display_work4(pixmap);
		break;
	default:
		break;
	}
}

void DlgNewProduction::pbtn_tab1_ok_clicked()
{
	auto result = QMessageBox::question(this, "确定", "你确定要清除所有文件吗");
	if (result == QMessageBox::Yes)
	{
		auto& globalStruct = GlobalStructData::getInstance();
		globalStruct.modelStorageManager->clear_temp();
		ui->tabWidget->setCurrentIndex(1);
		this->_info.currentTabIndex = 1;
	}
	else
	{
		return;
	}
}

void DlgNewProduction::flashImgCount()
{
	auto& modelStorageManager = GlobalStructData::getInstance().modelStorageManager;

	modelStorageManager->check_work_temp(modelStorageManager->imageSavePath);
	ui->label_tab3_tabImgCount1->setText(QString::number(modelStorageManager->work1_bad_count_));
	ui->label_tab3_tabImgCount2->setText(QString::number(modelStorageManager->work2_bad_count_));
	ui->label_tab3_tabImgCount3->setText(QString::number(modelStorageManager->work3_bad_count_));
	ui->label_tab3_tabImgCount4->setText(QString::number(modelStorageManager->work4_bad_count_));

	ui->label_tab4_tabImgCount1->setText(QString::number(modelStorageManager->work1_good_count_));
	ui->label_tab4_tabImgCount2->setText(QString::number(modelStorageManager->work2_good_count_));
	ui->label_tab4_tabImgCount3->setText(QString::number(modelStorageManager->work3_good_count_));
	ui->label_tab4_tabImgCount4->setText(QString::number(modelStorageManager->work4_good_count_));
}