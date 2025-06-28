#include "stdafx.h"

#include "ButtonScanner.h"
#include"NumberKeyboard.h"

#include"LoadingDialog.h"
#include"PicturesViewer.h"
#include"DlgProductSet.h"
#include"DlgProduceLineSet.h"
#include"GlobalStruct.h"
#include"ButtonUtilty.h"

#include"rqw_CameraObjectThreadZMotion.hpp"
#include"rqw_CameraObjectZMotion.hpp"
#include"hoec_CameraException.hpp"

#include<qdebug>
#include<QtConcurrent>
#include <future>
#include<QDir>
#include<QFileInfo>

void ButtonScanner::updateExposureTimeTrigger()
{
	// 获取窗口的当前宽度和高度
	auto windowWidth = ui->gBoix_ImageDisplay->width();
	auto windowHeight = ui->gBoix_ImageDisplay->height();

	// 计算目标区域的宽度和高度
	int targetWidth = static_cast<int>(windowWidth * exposureTimeTriggerWidthRatio);
	int targetHeight = static_cast<int>(windowHeight * exposureTimeTriggerRatio);

	// 计算目标区域的左上角位置，使其居中
	int targetX = (windowWidth - targetWidth) / 2;
	int targetY = (windowHeight - targetHeight) / 2;

	// 更新 targetArea
	exposureTimeTriggerArea = QRect(targetX, targetY, targetWidth, targetHeight);
}

void ButtonScanner::onExposureTimeTriggerAreaClicked()
{
	if (_dlgRealTimeImgDis) {
		_dlgRealTimeImgDis->hide();
	}
	auto& globalStructData = GlobalStructData::getInstance();
	auto isRuning = ui->rbtn_removeFunc->isChecked();
	if (!isRuning) {
		//ui->rbtn_debug->setChecked(false);
		auto& runningState = globalStructData.runningState;

		bool beforeSetExposureTimeIsDebug = false;
		if (runningState == RunningState::Debug)
		{
			beforeSetExposureTimeIsDebug = true;
		}

		runningState = RunningState::Monitor;
		//_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
		_dlgExposureTimeSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		_dlgExposureTimeSet->exec(); // 显示对话框
		//_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
		runningState = RunningState::Stop;
		if (beforeSetExposureTimeIsDebug)
		{
			rbtn_debug_checked(true); // 如果之前是调试模式，重新设置为调试模式
		}
	}
}

void ButtonScanner::mousePressEvent(QMouseEvent* event)
{
	updateExposureTimeTrigger();
	auto point = event->pos();
	// 检查鼠标点击是否在 targetArea 内
	if (exposureTimeTriggerArea.contains(event->pos())) {
		onExposureTimeTriggerAreaClicked(); // 调用目标函数
	}

	QMainWindow::mousePressEvent(event);
}

void ButtonScanner::resizeEvent(QResizeEvent* event)
{
	// 当窗口大小发生变化时，更新 targetArea
	updateExposureTimeTrigger();
	QMainWindow::resizeEvent(event);

	//// 获取 gBoix_ImageDisplay 的中心点
	//int displayX = ui->gBoix_ImageDisplay->x();
	//int displayY = ui->gBoix_ImageDisplay->y();
	//int displayWidth = ui->gBoix_ImageDisplay->width();
	//int displayHeight = ui->gBoix_ImageDisplay->height();

	//int centerX = displayX + displayWidth / 2;
	//int centerY = displayY + displayHeight / 2;

	//// 获取 label_lightBulb 的宽度和高度
	//int bulbWidth = label_lightBulb->width();
	//int bulbHeight = label_lightBulb->height();

	//// 计算 label_lightBulb 的新位置，使其中心对齐
	//int newX = centerX - bulbWidth / 2;
	//int newY = centerY - bulbHeight / 2;

	//// 设置 label_lightBulb 的位置
	//label_lightBulb->setGeometry(newX, newY, bulbWidth, bulbHeight);
}

ButtonScanner::ButtonScanner(QWidget* parent)
	: QMainWindow(parent)
	, ui(new Ui::ButtonScannerClass())
{
	GlobalStructData::getInstance().mainWindow = this; // 设置主窗口指针

	ui->setupUi(this);

	initializeComponents();
}

ButtonScanner::~ButtonScanner()
{
	destroyComponents();
	if (isShutdownByIO)
	{
		bool result = QProcess::startDetached("shutdown", QStringList() << "-s" << "-t" << "0");
		qDebug() << "Shutdown command started:" << result;
	}
	delete ui;
}

void ButtonScanner::set_radioButton()
{
	ui->rbtn_debug->setAutoExclusive(false);
	ui->rbtn_defect->setAutoExclusive(false);
	ui->rbtn_downLight->setAutoExclusive(false);
	ui->rbtn_ForAndAgainst->setAutoExclusive(false);
	ui->rbtn_removeFunc->setAutoExclusive(false);
	ui->rbtn_sideLight->setAutoExclusive(false);
	ui->rbtn_takePicture->setAutoExclusive(false);
	ui->rbtn_upLight->setAutoExclusive(false);
	ui->rbtn_strobe->setAutoExclusive(false);

	ui->rbtn_removeFunc->setAttribute(Qt::WA_TransparentForMouseEvents, true); // 禁止鼠标事件
	ui->rbtn_removeFunc->setFocusPolicy(Qt::NoFocus); // 禁止键盘焦点
}

void ButtonScanner::initializeComponents()
{
	_mark_thread = true;

	// 创建加载框
	LoadingDialog loadingDialog(this);
	loadingDialog.show();

	// 加载配置
	loadingDialog.updateMessage("正在加载配置...");
	QCoreApplication::processEvents(); // 保持 UI 响应
	read_config();

	build_modelStorageManager();

	// 构建 UI
	loadingDialog.updateMessage("正在构建界面...");
	QCoreApplication::processEvents();
	build_ui();
	this->setWindowFlags(Qt::FramelessWindowHint);

	// 连接信号与槽
	loadingDialog.updateMessage("正在建立信号与槽连接...");
	QCoreApplication::processEvents();
	build_connect();

	// 初始化运动控制
	loadingDialog.updateMessage("正在初始化运动控制...");
	QCoreApplication::processEvents();
	build_motion();

	// 停止所有轴
	loadingDialog.updateMessage("正在停止所有轴...");
	QCoreApplication::processEvents();
	//stop_all_axis();

	// 构建图像处理模块
	loadingDialog.updateMessage("正在构建图像处理模块...");
	QCoreApplication::processEvents();
	build_imageProcessorModule();

	// 构建相机
	loadingDialog.updateMessage("正在构建相机...");
	QCoreApplication::processEvents();
	build_camera();

	// 清理旧数据
	loadingDialog.updateMessage("正在清理旧数据...");
	QCoreApplication::processEvents();
	clear_olderSavedImage();

	// 构建图像保存引擎
	loadingDialog.updateMessage("正在构建图像保存引擎...");
	QCoreApplication::processEvents();
	build_imageSaveEngine();

	// 构建 IO 线程
	loadingDialog.updateMessage("正在构建 IO 线程...");
	QCoreApplication::processEvents();
	build_ioThread();

	// 启动监控
	loadingDialog.updateMessage("正在启动监控...");
	QCoreApplication::processEvents();
	start_monitor();

	// 构建位置线程
	loadingDialog.updateMessage("正在构建位置线程...");
	QCoreApplication::processEvents();
	build_locationThread();

	// 构建分离线程
	loadingDialog.updateMessage("正在构建后台线程...");
	QCoreApplication::processEvents();
	build_detachThread();

	QObject::connect(GlobalStructThread::getInstance().aiTrainModule.get(), &AiTrainModule::appRunLog,
		dlgNewProduction, &DlgNewProduction::appendAiTrainLog, Qt::QueuedConnection);
	QObject::connect(GlobalStructThread::getInstance().aiTrainModule.get(), &AiTrainModule::updateProgress,
		dlgNewProduction, &DlgNewProduction::updateProgress, Qt::QueuedConnection);
	QObject::connect(GlobalStructThread::getInstance().aiTrainModule.get(), &AiTrainModule::updateTrainTitle,
		dlgNewProduction, &DlgNewProduction::updateProgressTitle, Qt::QueuedConnection);
	QObject::connect(GlobalStructThread::getInstance().aiTrainModule.get(), &AiTrainModule::updateTrainState,
		dlgNewProduction, &DlgNewProduction::updateTrainState, Qt::QueuedConnection);
	QObject::connect(dlgNewProduction, &DlgNewProduction::cancelTrain,
		GlobalStructThread::getInstance().aiTrainModule.get(), &AiTrainModule::cancelTrain);
	QObject::connect(GlobalStructThread::getInstance().detachUtiltyThread.get(), &DetachUtiltyThread::showDlgWarn,
		this, &ButtonScanner::showDlgWarn, Qt::QueuedConnection);
	QObject::connect(GlobalStructThread::getInstance().detachUtiltyThread.get(), &DetachUtiltyThread::workTriggerError,
		this, &ButtonScanner::workTriggerError, Qt::QueuedConnection);
	QObject::connect(GlobalStructThread::getInstance().detachUtiltyThread.get(), &DetachUtiltyThread::closeTakePictures,
		this, &ButtonScanner::closeTakePictures, Qt::QueuedConnection);
	QObject::connect(GlobalStructThread::getInstance().detachUtiltyThread.get(), &DetachUtiltyThread::shutdownComputer,
		this, &ButtonScanner::shutdownComputerTrigger, Qt::QueuedConnection);
	auto mainWindowConfig = GlobalStructData::getInstance().mainWindowConfig;

	//初始化光源
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motionPtr->SetIOOut(ControlLines::upLightOut, mainWindowConfig.upLight);
	motionPtr->SetIOOut(ControlLines::downLightOut, mainWindowConfig.downLight);
	motionPtr->SetIOOut(ControlLines::sideLightOut, mainWindowConfig.sideLight);
	motionPtr->SetIOOut(ControlLines::strobeLightOut, mainWindowConfig.strobeLight);

	// 隐藏加载框
	loadingDialog.close();
}

void ButtonScanner::destroyComponents()
{
	// 创建加载框
	LoadingDialog loadingDialog(this);
	loadingDialog.show();

	// 停止线程标志
	loadingDialog.updateMessage("正在停止线程...");
	QCoreApplication::processEvents();
	_mark_thread = false;

	// 销毁分离线程
	loadingDialog.updateMessage("正在销毁后台线程...");
	QCoreApplication::processEvents();
	auto& globalStructThread = GlobalStructThread::getInstance();
	globalStructThread.destroyDetachThread();

	// 停止所有轴
	loadingDialog.updateMessage("正在停止所有轴...");
	QCoreApplication::processEvents();
	stop_all_axis();

	// 销毁相机
	loadingDialog.updateMessage("正在销毁相机...");
	QCoreApplication::processEvents();
	auto& globalStructData = GlobalStructData::getInstance();
	globalStructData.destroyCamera();

	// 销毁图像处理模块
	loadingDialog.updateMessage("正在销毁图像处理模块...");
	QCoreApplication::processEvents();
	globalStructData.destroyImageProcessingModule();

	// 销毁图像保存引擎
	loadingDialog.updateMessage("正在销毁图像保存引擎...");
	QCoreApplication::processEvents();
	globalStructData.destroyImageSaveEngine();

	destroy_modelStorageManager();
	// 保存配置
	loadingDialog.updateMessage("正在保存配置...");
	QCoreApplication::processEvents();
	//关机自动关闭debug模式和采图
	globalStructData.mainWindowConfig.isDebugMode = false;
	globalStructData.mainWindowConfig.isTakePictures = false;
	globalStructData.saveConfig();

	// 删除 UI
	loadingDialog.updateMessage("正在清理界面...");
	QCoreApplication::processEvents();

	// 隐藏加载框
	loadingDialog.close();
}

void ButtonScanner::build_ui()
{
	label_lightBulb = new QLabel(this);
	dlgWarn = new DlgWarn(this);

	//Set RadioButton ,make sure these can be checked at the same time
	read_image();
	set_radioButton();
	build_mainWindowData();
	build_dlgProduceLineSet();
	build_dlgProductSet();
	build_dlgExposureTimeSet();
	build_dlgNewProduction();
	build_picturesViewer();
	_dlgShutdownWarn = new DlgShutdownWarn(this);
	build_dlgModelManager();
	this->labelClickable_title = new rw::rqw::ClickableLabel(this);
	labelWarning = new rw::rqw::LabelWarning(this);
	ui->gBox_warningInfo->layout()->replaceWidget(ui->label_warningInfo, labelWarning);
	delete ui->label_warningInfo;

	labelClickable_title->setText(ui->label_title->text());
	labelClickable_title->setStyleSheet(ui->label_title->styleSheet());
	ui->hLayout_title->replaceWidget(ui->label_title, labelClickable_title);
	delete ui->label_title;

	labelVersionInfo = new rw::rqw::ClickableLabel(this);
	ui->gBox_VersionInfo->layout()->replaceWidget(ui->label_VersionInfo, labelVersionInfo);
	delete ui->label_VersionInfo;
	labelVersionInfo->setText(VersionInfo::Version);

	_dlgVersion = new DlgVersion(this);
	QString versionFilePath = QCoreApplication::applicationDirPath() + QDir::separator() + "Version.txt";
	_dlgVersion->loadVersionPath(versionFilePath);

	QObject::connect(_dlgModelManager, &DlgModelManager::updateExposureTime
		, this, &ButtonScanner::updateExposureTimeValueOnDlg);
	QObject::connect(_dlgModelManager, &DlgModelManager::checkPosiviveRadioButtonCheck
		, this, &ButtonScanner::checkPosiviveRadioButtonCheck);

	ui->cBox_isDisplayRec->setVisible(false);
	ui->cBox_isDisplayText->setVisible(false);
	//Deprecated
	ui->pbtn_beltSpeed->setVisible(false);
	ui->rbtn_strobe->setVisible(false);

	imgDis1 = new rw::rqw::ClickableLabel(this);
	imgDis1->setSizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);

	imgDis2 = new rw::rqw::ClickableLabel(this);
	imgDis2->setSizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);

	imgDis3 = new rw::rqw::ClickableLabel(this);
	imgDis3->setSizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);

	imgDis4 = new rw::rqw::ClickableLabel(this);
	imgDis4->setSizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);


	ui->gBoix_ImageDisplay->layout()->replaceWidget(ui->label_imgDisplay, imgDis1);
	ui->gBoix_ImageDisplay->layout()->replaceWidget(ui->label_imgDisplay_2, imgDis2);
	ui->gBoix_ImageDisplay->layout()->replaceWidget(ui->label_imgDisplay_3, imgDis3);
	ui->gBoix_ImageDisplay->layout()->replaceWidget(ui->label_imgDisplay_4, imgDis4);

	delete ui->label_imgDisplay;
	delete ui->label_imgDisplay_2;
	delete ui->label_imgDisplay_3;
	delete ui->label_imgDisplay_4;

	QObject::connect(imgDis1, &rw::rqw::ClickableLabel::clicked
		, this, &ButtonScanner::imgDis1_clicked);
	QObject::connect(imgDis2, &rw::rqw::ClickableLabel::clicked
		, this, &ButtonScanner::imgDis2_clicked);
	QObject::connect(imgDis3, &rw::rqw::ClickableLabel::clicked
		, this, &ButtonScanner::imgDis3_clicked);
	QObject::connect(imgDis4, &rw::rqw::ClickableLabel::clicked
		, this, &ButtonScanner::imgDis4_clicked);

	_dlgRealTimeImgDis = new DlgRealTimeImgDis(this);
	_dlgRealTimeImgDis->setMonitorValue(&_isRealTimeDis);
	_dlgRealTimeImgDis->setMonitorDisImgIndex(&_currentRealTimeDisIndex);
}

void ButtonScanner::read_image()
{
	QString imagePath = ":/ButtonScanner/image/lightBulb.png";
	QPixmap pixmap(imagePath);

	if (pixmap.isNull()) {
		QMessageBox::critical(this, "Error", "无法加载图片。");
		return;
	}
	label_lightBulb->setPixmap(pixmap.scaled(label_lightBulb->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ButtonScanner::build_mainWindowData()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& mainWindowConfig = globalStruct.mainWindowConfig;
	ui->label_produceTotalValue->setText(QString::number(mainWindowConfig.totalProduction));
	ui->label_wasteProductsValue->setText(QString::number(mainWindowConfig.totalWaste));
	ui->label_productionYieldValue->setText(QString::number(mainWindowConfig.passRate) + QString(" %"));
	ui->rbtn_debug->setChecked(mainWindowConfig.isDebugMode);
	ui->rbtn_takePicture->setChecked(mainWindowConfig.isTakePictures);
	ui->rbtn_removeFunc->setChecked(mainWindowConfig.isEliminating);
	ui->label_removeRate->setText(QString::number(mainWindowConfig.scrappingRate) + QString(" /min"));
	ui->rbtn_upLight->setChecked(mainWindowConfig.upLight);
	ui->rbtn_downLight->setChecked(mainWindowConfig.downLight);
	ui->rbtn_sideLight->setChecked(mainWindowConfig.sideLight);
	ui->rbtn_defect->setChecked(mainWindowConfig.isDefect);
	ui->rbtn_strobe->setChecked(mainWindowConfig.strobeLight);
	ui->rbtn_ForAndAgainst->setChecked(mainWindowConfig.isPositive);
	ui->pbtn_beltSpeed->setText(QString::number(globalStruct.dlgProduceLineSetConfig.motorSpeed));
}

void ButtonScanner::build_dlgProduceLineSet()
{
	this->_dlgProduceLineSet = new DlgProduceLineSet(this);
}

void ButtonScanner::build_dlgProductSet()
{
	this->_dlgProductSet = new DlgProductSet(this);
}

void ButtonScanner::build_dlgExposureTimeSet()
{
	this->_dlgExposureTimeSet = new DlgExposureTimeSet(this);
	//设置对话框的初始位置
	int x = this->x() + (this->width() - _dlgExposureTimeSet->width()) / 2;
	int y = this->y() + (this->height() - _dlgExposureTimeSet->height()) / 2;
	_dlgExposureTimeSet->move(x, y);
}

void ButtonScanner::stop_all_axis()
{
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motionPtr->StopAllAxis();
	motionPtr->SetIOOut(ControlLines::motoPowerOut, false);
	motionPtr->SetIOOut(ControlLines::warnRedOut, false);
	motionPtr->SetIOOut(ControlLines::warnGreenOut, false);
}

void ButtonScanner::build_dlgNewProduction()
{
	this->dlgNewProduction = new DlgNewProduction(this);
}

void ButtonScanner::build_modelStorageManager()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.modelStorageManager = std::make_unique<ModelStorageManager>(this);
	globalStruct.modelStorageManager->setRootPath(globalPath.modelStorageManagerRootPath);
}

void ButtonScanner::destroy_modelStorageManager()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.modelStorageManager.reset();
}

void ButtonScanner::build_picturesViewer()
{
	_picturesViewer = new PictureViewerThumbnails(this);
	_picturesViewer->setSize({ 100,100 });
	_picturesViewer->setThumbnailCacheCapacity(1000);
}

void ButtonScanner::build_dlgModelManager()
{
	_dlgModelManager = new DlgModelManager(this);
}

void ButtonScanner::build_connect()
{
	QObject::connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_exit_clicked);

	QObject::connect(ui->pbtn_set, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_set_clicked);

	QObject::connect(ui->pbtn_newProduction, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_newProduction_clicked);

	QObject::connect(ui->pbtn_beltSpeed, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_beltSpeed_clicked);

	QObject::connect(ui->pbtn_score, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_score_clicked);

	QObject::connect(ui->rbtn_debug, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_debug_checked);

	QObject::connect(ui->rbtn_takePicture, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_takePicture_checked);

	QObject::connect(ui->rbtn_removeFunc, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_removeFunc_checked);

	QObject::connect(ui->rbtn_upLight, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_upLight_checked);

	QObject::connect(ui->rbtn_sideLight, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_sideLight_checked);

	QObject::connect(ui->rbtn_downLight, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_downLight_checked);

	QObject::connect(ui->rbtn_defect, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_defect_checked);

	QObject::connect(ui->rbtn_strobe, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_strobe_checked);

	QObject::connect(ui->rbtn_ForAndAgainst, &QRadioButton::clicked,
		this, &ButtonScanner::rbtn_forAndAgainst_checked);

	QObject::connect(ui->pbtn_resetProduct, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_resetProduct_clicked);

	QObject::connect(ui->pbtn_openSaveLocation, &QPushButton::clicked,
		this, &ButtonScanner::pbtn_openSaveLocation_clicked);

	QObject::connect(this->labelClickable_title, &rw::rqw::ClickableLabel::clicked,
		this, &ButtonScanner::labelClickable_title_clicked);

	QObject::connect(&GlobalStructData::getInstance(), &GlobalStructData::updateLightState,
		this, &ButtonScanner::onUpdateLightStateUi);

	QObject::connect(dlgWarn, &DlgWarn::isProcess,
		this, &ButtonScanner::dlgWarningAccept);

	QObject::connect(ui->cBox_isDisplayRec, &QCheckBox::clicked,
		this, &ButtonScanner::cBox_isDisplayRec_checked);

	QObject::connect(ui->cBox_isDisplayText, &QCheckBox::clicked,
		this, &ButtonScanner::cBox_isDisplayText_checked);

	QObject::connect(labelVersionInfo, &rw::rqw::ClickableLabel::clicked,
		this, &ButtonScanner::labelVersion_clicked);
}

void ButtonScanner::read_config()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.buildConfigManager(rw::oso::StorageType::Xml);

	read_config_mainWindowConfig();
	read_config_productSetConfig();
	read_config_produceLineConfig();
	read_config_exposureTimeSetConfig();
	read_config_hideScoreSet();
	read_config_warningManagerConfig();
	read_config_warningIOSetConfig();
}

void ButtonScanner::read_config_mainWindowConfig()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QString mainWindowFilePathFull = globalPath.configRootPath + "mainWindowConfig.xml";
	QFileInfo mainWindowFile(mainWindowFilePathFull);

	globalStruct.mainWindowFilePath = mainWindowFilePathFull;

	if (!mainWindowFile.exists()) {
		QDir configDir = QFileInfo(mainWindowFilePathFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(mainWindowFilePathFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件mainWindowConfig.xml");
		}
		globalStruct.mainWindowConfig = cdm::ButtonScannerMainWindow();
		globalStruct.saveMainWindowConfig();
		return;
	}
	else {
		globalStruct.ReadMainWindowConfig();
	}
}

void ButtonScanner::read_config_warningManagerConfig()
{
	auto& globalStruct = GlobalStructData::getInstance();
	QString warningManagerFilePathFull = globalPath.configRootPath + "warningManagerConfig.xml";
	QFileInfo warningManagerFile(warningManagerFilePathFull);
	globalStruct.warningManagerFilePath = warningManagerFilePathFull;
	if (!warningManagerFile.exists()) {
		QDir configDir = QFileInfo(warningManagerFilePathFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(warningManagerFilePathFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件warningManagerConfig.xml");
		}
		globalStruct.dlgWarningManagerConfig = rw::cdm::ButtonScannerDlgWarningManager();
		globalStruct.saveWarningManagerConfig();
		return;
	}
	else {
		globalStruct.ReadWarningManagerConfig();
	}
}

void ButtonScanner::read_config_produceLineConfig()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QString dlgProduceLineSetFilePathFull = globalPath.configRootPath + "dlgProduceLineSetConfig.xml";
	QFileInfo dlgProduceLineSetFile(dlgProduceLineSetFilePathFull);

	globalStruct.dlgProduceLineSetFilePath = dlgProduceLineSetFilePathFull;

	if (!dlgProduceLineSetFile.exists()) {
		QDir configDir = QFileInfo(dlgProduceLineSetFilePathFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(dlgProduceLineSetFilePathFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件dlgProduceLineSetConfig.xml");
		}
		globalStruct.dlgProduceLineSetConfig = cdm::ButtonScannerProduceLineSet();
		globalStruct.saveDlgProduceLineSetConfig();
		return;
	}
	else {
		globalStruct.ReadDlgProduceLineSetConfig();
	}
	globalStruct.dlgProduceLineSetConfig.takeNgPictures = true;
	globalStruct.dlgProduceLineSetConfig.takeMaskPictures = true;
	globalStruct.dlgProduceLineSetConfig.takeOkPictures = true;
}

void ButtonScanner::read_config_productSetConfig()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QString dlgProductSetFilePathFull = globalPath.configRootPath + "dlgProdutSetConfig.xml";
	QFileInfo dlgProductSetFile(dlgProductSetFilePathFull);

	globalStruct.dlgProductSetFilePath = dlgProductSetFilePathFull;

	if (!dlgProductSetFile.exists()) {
		QDir configDir = QFileInfo(dlgProductSetFilePathFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(dlgProductSetFilePathFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件dlgProdutSetConfig.xml");
		}
		globalStruct.dlgProductSetConfig = cdm::ButtonScannerDlgProductSet();
		globalStruct.saveDlgProductSetConfig();
		return;
	}
	else {
		globalStruct.ReadDlgProductSetConfig();
	}
}

void ButtonScanner::read_config_exposureTimeSetConfig()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QString exposureTimeSetConfigFilePathFull = globalPath.configRootPath + "exposureTimeSetConfig.xml";
	QFileInfo dlgProductSetFile(exposureTimeSetConfigFilePathFull);

	globalStruct.dlgExposureTimeSetFilePath = exposureTimeSetConfigFilePathFull;

	if (!dlgProductSetFile.exists()) {
		QDir configDir = QFileInfo(exposureTimeSetConfigFilePathFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(exposureTimeSetConfigFilePathFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件exposureTimeSetConfig.xml");
		}
		globalStruct.dlgExposureTimeSetConfig = cdm::ButtonScannerDlgExposureTimeSet();
		globalStruct.saveDlgExposureTimeSetConfig();
		return;
	}
	else {
		globalStruct.ReadDlgExposureTimeSetConfig();
	}
}

void ButtonScanner::read_config_hideScoreSet()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QString dlgHideScoreSetFilePathFull = globalPath.configRootPath + "dlgHideScoreSet.xml";
	QFileInfo dlgHideScoreSetFile(dlgHideScoreSetFilePathFull);

	globalStruct.dlgHideScoreSetPath = dlgHideScoreSetFilePathFull;

	if (!dlgHideScoreSetFile.exists()) {
		QDir configDir = QFileInfo(dlgHideScoreSetFilePathFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(dlgHideScoreSetFilePathFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件mainWindowConfig.xml");
		}
		globalStruct.dlgHideScoreSetConfig = cdm::DlgHideScoreSet();
		globalStruct.saveDlgHideScoreSetConfig();
		return;
	}
	else {
		globalStruct.ReadDlgHideScoreSetConfig();
	}
}

void ButtonScanner::read_config_warningIOSetConfig()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QString dlgWarningIOSetConfigFull = globalPath.configRootPath + "warningIOSetConfig.xml";
	QFileInfo dlgWarningIOSetConfigFile(dlgWarningIOSetConfigFull);

	globalStruct.warningIOSetConfigPath = dlgWarningIOSetConfigFull;

	if (!dlgWarningIOSetConfigFile.exists()) {
		QDir configDir = QFileInfo(dlgWarningIOSetConfigFull).absoluteDir();
		if (!configDir.exists()) {
			configDir.mkpath(".");
		}
		QFile file(dlgWarningIOSetConfigFull);
		if (file.open(QIODevice::WriteOnly)) {
			file.close();
		}
		else {
			QMessageBox::critical(this, "Error", "无法创建配置文件warningIOSetConfig.xml");
		}
		globalStruct.warningIOSetConfig = cdm::WarningIOSetConfig();
		globalStruct.saveWarningIOSetConfig();
		return;
	}
	else {
		globalStruct.readWarningIOSetConfig();
	}
}

void ButtonScanner::build_imageSaveEngine()
{
	QDir dir;
	QString imageSavePath = globalPath.imageSaveRootPath;
	//清理旧的数据

	//获取当前日期并设置保存路径
	QString currentDate = QDate::currentDate().toString("yyyy_MM_dd");
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.buildImageSaveEngine();
	QString imageSaveEnginePath = imageSavePath + currentDate;

	QString imagesFilePathFilePathFull = dir.absoluteFilePath(imageSaveEnginePath);
	globalStruct.imageSaveEngine->setRootPath(imagesFilePathFilePathFull);
	globalStruct.imageSaveEngine->startEngine();
}

void ButtonScanner::clear_olderSavedImage()
{
	QString imageSavePath = globalPath.imageSaveRootPath;
	QVector<QString> sortedFolders;

	// 打开指定路径
	QDir dir(imageSavePath);
	if (!dir.exists()) {
		return;
	}

	// 设置过滤器，只获取文件夹
	dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);

	// 获取文件夹列表
	QFileInfoList folderList = dir.entryInfoList();

	// 存储文件夹及其日期
	QVector<QPair<QDate, QString>> dateFolderPairs;

	for (const QFileInfo& folderInfo : folderList) {
		QString folderName = folderInfo.fileName();

		// 尝试解析文件夹名称为日期
		QDate folderDate = QDate::fromString(folderName, "yyyy_MM_dd");
		if (folderDate.isValid()) {
			dateFolderPairs.append(qMakePair(folderDate, folderName));
		}
	}

	// 按日期排序
	std::sort(dateFolderPairs.begin(), dateFolderPairs.end(),
		[](const QPair<QDate, QString>& a, const QPair<QDate, QString>& b) {
			return a.first < b.first;
		});

	// 提取排序后的文件夹名称
	for (const auto& pair : dateFolderPairs) {
		sortedFolders.append(pair.second);
	}

	// 删除超过7天的文件夹
	QDate currentDate = QDate::currentDate();
	for (const QString& folderName : sortedFolders) {
		QDate folderDate = QDate::fromString(folderName, "yyyy_MM_dd");
		if (folderDate.isValid() && folderDate < currentDate.addDays(-7)) {
			QString folderPath = imageSavePath + folderName;
			QDir(folderPath).removeRecursively();
		}
	}
}

void ButtonScanner::build_camera()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.cameraIp1 = "11";
	globalStruct.cameraIp2 = "12";
	globalStruct.cameraIp3 = "13";
	globalStruct.cameraIp4 = "14";

	auto build1Result = globalStruct.buildCamera1();
	updateCameraLabelState(1, build1Result);
	if (!build1Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机1连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm1;
		info.type = rw::rqw::WarningType::Error;
		labelWarning->addWarning(info);
	}

	auto build2Result = globalStruct.buildCamera2();
	updateCameraLabelState(2, build2Result);
	if (!build2Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机2连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm2;
		info.type = rw::rqw::WarningType::Error;
		labelWarning->addWarning(info);
	}

	auto build3Result = globalStruct.buildCamera3();
	updateCameraLabelState(3, build3Result);
	if (!build3Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机3连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm3;
		info.type = rw::rqw::WarningType::Error;
		labelWarning->addWarning(info);
	}

	auto build4Result = globalStruct.buildCamera4();
	updateCameraLabelState(4, build4Result);
	if (!build4Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机4连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm4;
		info.type = rw::rqw::WarningType::Error;
		labelWarning->addWarning(info);
	}

	_dlgExposureTimeSet->ResetCamera(); //启动设置相机为默认状态
}

void ButtonScanner::build_imageProcessorModule()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QDir dir;

	QString enginePathFull = globalPath.modelRootPath + globalPath.engineSeg;
	QString onnxEnginePathFull1 = globalPath.modelRootPath + globalPath.onnxRuntime1;
	QString onnxEnginePathFull2 = globalPath.modelRootPath + globalPath.onnxRuntime2;
	QString onnxEnginePathFull3 = globalPath.modelRootPath + globalPath.onnxRuntime3;
	QString onnxEnginePathFull4 = globalPath.modelRootPath + globalPath.onnxRuntime4;

	QFileInfo engineFile(enginePathFull);
	QFileInfo onnxEngineFile1(onnxEnginePathFull1);
	QFileInfo onnxEngineFile2(onnxEnginePathFull2);
	QFileInfo onnxEngineFile3(onnxEnginePathFull3);
	QFileInfo onnxEngineFile4(onnxEnginePathFull4);

	if (!engineFile.exists() || !onnxEngineFile1.exists() || !onnxEngineFile2.exists() || !onnxEngineFile3.exists() || !onnxEngineFile4.exists()) {
		QMessageBox::critical(this, "Error", "Engine file or Name file does not exist. The application will now exit.");
		QApplication::quit();
		return;
	}

	globalStruct.enginePath = enginePathFull;
	globalStruct.onnxEngineOOPath1 = onnxEnginePathFull1;
	globalStruct.onnxEngineOOPath2 = onnxEnginePathFull2;
	globalStruct.onnxEngineOOPath3 = onnxEnginePathFull3;
	globalStruct.onnxEngineOOPath4 = onnxEnginePathFull4;

	globalStruct.buildImageProcessingModule(2);

	////连接界面显示和图像处理模块
	QObject::connect(globalStruct.imageProcessingModule1.get(), &ImageProcessingModule::imageReady,
		this, &ButtonScanner::onCamera1Display, Qt::DirectConnection);
	QObject::connect(globalStruct.imageProcessingModule1.get(), &ImageProcessingModule::imgForDlgNewProduction,
		this->dlgNewProduction, &DlgNewProduction::img_display_work, Qt::DirectConnection);

	QObject::connect(globalStruct.imageProcessingModule2.get(), &ImageProcessingModule::imageReady,
		this, &ButtonScanner::onCamera2Display, Qt::DirectConnection);
	QObject::connect(globalStruct.imageProcessingModule2.get(), &ImageProcessingModule::imgForDlgNewProduction,
		this->dlgNewProduction, &DlgNewProduction::img_display_work, Qt::DirectConnection);

	QObject::connect(globalStruct.imageProcessingModule3.get(), &ImageProcessingModule::imageReady,
		this, &ButtonScanner::onCamera3Display, Qt::DirectConnection);
	QObject::connect(globalStruct.imageProcessingModule3.get(), &ImageProcessingModule::imgForDlgNewProduction,
		this->dlgNewProduction, &DlgNewProduction::img_display_work, Qt::DirectConnection);
	QObject::connect(globalStruct.imageProcessingModule4.get(), &ImageProcessingModule::imageReady,
		this, &ButtonScanner::onCamera4Display, Qt::DirectConnection);
	QObject::connect(globalStruct.imageProcessingModule4.get(), &ImageProcessingModule::imgForDlgNewProduction,
		this->dlgNewProduction, &DlgNewProduction::img_display_work, Qt::DirectConnection);
}

void ButtonScanner::start_monitor()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.startMonitor();
}

void ButtonScanner::build_motion()
{
	auto& globalStruct = GlobalStructData::getInstance();

	//获取Zmotion
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;

	//下面通过motionPtr进行操作
	bool state = motionPtr.get()->OpenBoard((char*)"192.168.0.11");

	if (state) {
		motionPtr->SetLocationZero(0);
		motionPtr->SetLocationZero(1);
		motionPtr->SetLocationZero(2);
		motionPtr->SetAxisType(1, 3);
		motionPtr->SetAxisType(2, 3);
		motionPtr->SetAxisPulse(1, globalStruct.dlgProduceLineSetConfig.codeWheel);
		motionPtr->SetAxisPulse(2, globalStruct.dlgProduceLineSetConfig.codeWheel);
		updateCardLabelState(true);
	}
	else
	{
		rw::rqw::WarningInfo info;
		info.message = "运动控制器连接失败";
		info.warningId = WarningId::csportControlAlarm;
		info.type = rw::rqw::WarningType::Error;
		labelWarning->addWarning(info);
		updateCardLabelState(false);
	}
}

void ButtonScanner::build_locationThread()
{
	//线程内部
	QFuture<void>  m_monitorFuture = QtConcurrent::run([this]() {
		while (_mark_thread)
		{
			auto& globalStruct = GlobalStructData::getInstance();
			auto& blowTime = globalStruct.dlgProductSetConfig.blowTime;
			//获得位置数据

			//1,3相机
			float lacation1 = 0;
			//2，4相机
			float lacation2 = 0;
			//获取两个位置
			zwy::scc::GlobalMotion::getInstance().motionPtr.get()->GetAxisLocation(2, lacation1);
			zwy::scc::GlobalMotion::getInstance().motionPtr.get()->GetAxisLocation(1, lacation2);

			if (lacation1 < 0)
			{
				lacation1 = -lacation1;
			}

			if (lacation2 < 0)
			{
				lacation2 = -lacation2;
			}

			{
				auto& work1 = GlobalStructData::getInstance().productPriorityQueue1;

				double tifeishijian1 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowTime1 + blowTime;
				double tifeijuli1 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowDistance1;

				float olderlacation1 = 0;
				bool isGet = work1.tryGetMin(olderlacation1);

				if (isGet != false && (abs(lacation1 - olderlacation1) > tifeijuli1))
				{
					work1.tryPopMin(olderlacation1);

					//吹气
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::blowLine1.axis, ControlLines::blowLine1.ioNum, true, tifeishijian1);
				}
			}
			{
				auto& work2 = GlobalStructData::getInstance().productPriorityQueue2;

				double tifeishijian2 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowTime2 + blowTime;
				double tifeijuli2 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowDistance2;

				float olderlacation2 = 0;
				bool isGet = work2.tryGetMin(olderlacation2);
				if (olderlacation2 != false && (abs(lacation2 - olderlacation2) > tifeijuli2))
				{
					work2.tryPopMin(olderlacation2);

					//吹气
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::blowLine2.axis, ControlLines::blowLine2.ioNum, true, tifeishijian2);
				}
			}

			{
				auto& work3 = GlobalStructData::getInstance().productPriorityQueue3;

				double tifeishijian3 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowTime3 + blowTime;
				double tifeijuli3 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowDistance3;

				float olderlacation3 = 0;
				bool isGet = work3.tryGetMin(olderlacation3);
				if (isGet != false && abs(lacation1 - olderlacation3) > tifeijuli3)
				{
					work3.tryPopMin(olderlacation3);

					//吹气
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::blowLine3.axis, ControlLines::blowLine3.ioNum, true, tifeishijian3);
				}
			}

			{
				auto& work4 = GlobalStructData::getInstance().productPriorityQueue4;

				double tifeishijian4 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowTime4 + blowTime;
				double tifeijuli4 = GlobalStructData::getInstance().dlgProduceLineSetConfig.blowDistance4;

				float olderlacation4 = 0;
				bool isGet = work4.tryGetMin(olderlacation4);
				if (isGet != false && abs(lacation2 - olderlacation4) > tifeijuli4)
				{
					work4.tryPopMin(olderlacation4);

					//吹气
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::blowLine4.axis, ControlLines::blowLine4.ioNum, true, tifeishijian4);
				}
			}
		}
		});
}

void ButtonScanner::build_ioThread()
{
	//线程内部
	QtConcurrent::run([this]() {
		auto& globalStruct = GlobalStructData::getInstance();
		//获取Zmotion
		auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;

		while (_mark_thread)
		{
			bool state = false;
			state = motionPtr->GetIOIn(ControlLines::stopIn);
			//急停

			if (state == true)
			{
				globalStruct.runningState = RunningState::Stop;
				QMetaObject::invokeMethod(qApp, [this, state]
					{
						ui->rbtn_removeFunc->setChecked(false);
						ui->rbtn_debug->setChecked(false);
						rbtn_debug_checked(false);
						label_lightBulb->setVisible(true);
					});
				// pidaimove->stop();
				motionPtr->StopAllAxis();
				motionPtr->SetIOOut(ControlLines::motoPowerOut, false);

				motionPtr->SetIOOut(ControlLines::warnGreenOut, false);
			}
			else
			{
				//开始按钮
				bool state = false;
				state = motionPtr->GetIOIn(ControlLines::startIn);
				//启动程序
				if (state == true)
				{
					if (dlgNewProduction->_info.isActivate == false)
					{
						globalStruct.runningState = RunningState::OpenRemoveFunc;
						globalStruct.imageProcessingModule1->clearLargeRGBList();
						globalStruct.imageProcessingModule2->clearLargeRGBList();
						globalStruct.imageProcessingModule3->clearLargeRGBList();
						globalStruct.imageProcessingModule4->clearLargeRGBList();
						QMetaObject::invokeMethod(qApp, [this, state]
							{
								_dlgExposureTimeSet->ResetCamera();
								ui->rbtn_removeFunc->setChecked(true);
								ui->rbtn_debug->setChecked(false);
								label_lightBulb->setVisible(false);
								ui->cBox_isDisplayRec->setVisible(false);
								ui->cBox_isDisplayText->setVisible(false);
							});
					}
					//所有电机上电
					QtConcurrent::run([this, &motionPtr]() {
						QThread::msleep(500);
						motionPtr->SetIOOut(ControlLines::motoPowerOut, true);
						//启动电机
						motionPtr->SetAxisType(ControlLines::beltAsis, 1);
						double unit = GlobalStructData::getInstance().dlgProduceLineSetConfig.pulseFactor;
						motionPtr->SetAxisPulse(ControlLines::beltAsis, unit);
						double acc = GlobalStructData::getInstance().dlgProduceLineSetConfig.accelerationAndDeceleration;
						motionPtr->SetAxisAcc(ControlLines::beltAsis, acc);
						motionPtr->SetAxisDec(ControlLines::beltAsis, acc);
						double speed = GlobalStructData::getInstance().dlgProduceLineSetConfig.motorSpeed;
						motionPtr->SetAxisRunSpeed(ControlLines::beltAsis, speed);
						// pidaimove->start(100);
						motionPtr->SetAxisRun(ControlLines::beltAsis, -1);
						motionPtr->SetIOOut(ControlLines::warnGreenOut, true);
						});
				}
				//停止点
				state = motionPtr->GetIOIn(ControlLines::stopIn);
				if (state)
				{
					globalStruct.runningState = RunningState::Stop;
					QMetaObject::invokeMethod(qApp, [this, state]
						{
							ui->rbtn_removeFunc->setChecked(false);
							ui->rbtn_debug->setChecked(false);
							rbtn_debug_checked(false);
							label_lightBulb->setVisible(false);
						});
					motionPtr->StopAllAxis();
					motionPtr->SetIOOut(ControlLines::motoPowerOut, false);
					motionPtr->SetIOOut(ControlLines::warnGreenOut, false);
				}

				auto qiya = motionPtr->GetIOIn(ControlLines::airWarnIn);
				if (qiya == true) {
					QMetaObject::invokeMethod(qApp, [this, state]
						{
							rw::rqw::WarningInfo info;
							info.message = "气压不正常";
							info.type = rw::rqw::WarningType::Error;
							info.warningId = WarningId::cairPressureAlarm;
							labelWarning->addWarning(info, true);
						});
				}
			}

			QThread::msleep(200);
		}
		});
}

void ButtonScanner::build_detachThread()
{
	auto& globalStruct = GlobalStructThread::getInstance();
	globalStruct.buildDetachThread();
	QObject::connect(globalStruct.detachUtiltyThread.get(), &DetachUtiltyThread::updateStatisticalInfo,
		this, &ButtonScanner::updateStatisticalInfoUI, Qt::QueuedConnection);
	QObject::connect(globalStruct.detachUtiltyThread.get(), &DetachUtiltyThread::addWarningInfo,
		this, &ButtonScanner::onAddWarningInfo, Qt::QueuedConnection);

	QObject::connect(globalStruct.monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::updateCameraLabelState,
		this, &ButtonScanner::updateCameraLabelState, Qt::QueuedConnection);
	QObject::connect(globalStruct.monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::updateCardLabelState,
		this, &ButtonScanner::updateCardLabelState, Qt::QueuedConnection);
	QObject::connect(globalStruct.monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::addWarningInfo,
		this, &ButtonScanner::onAddWarningInfo, Qt::QueuedConnection);

	globalStruct.detachUtiltyThread->warningLabel = labelWarning;
}

void ButtonScanner::showEvent(QShowEvent* event)
{
	int displayX = ui->gBoix_ImageDisplay->x();
	int displayY = ui->gBoix_ImageDisplay->y();
	int displayWidth = ui->gBoix_ImageDisplay->width();
	int displayHeight = ui->gBoix_ImageDisplay->height();

	int centerX = displayX + displayWidth / 2;
	int centerY = displayY + displayHeight / 2;

	// 获取 label_lightBulb 的宽度和高度
	int bulbWidth = label_lightBulb->width();
	int bulbHeight = label_lightBulb->height();

	// 计算 label_lightBulb 的新位置，使其中心对齐
	int newX = centerX - bulbWidth / 2;
	int newY = centerY - bulbHeight / 2;

	// 设置 label_lightBulb 的位置
	label_lightBulb->setGeometry(newX + 30, newY, bulbWidth, bulbHeight);
}

QImage ButtonScanner::cvMatToQImage(const cv::Mat& mat)
{
	if (mat.type() == CV_8UC1) {
		return QImage(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_Grayscale8);
	}
	else if (mat.type() == CV_8UC3) {
		return QImage(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_RGB888).rgbSwapped();
	}
	else if (mat.type() == CV_8UC4) {
		return QImage(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_RGBA8888);
	}
	else {
		return QImage();
	}
}

void ButtonScanner::onUpdateLightStateUi(size_t index, bool state)
{
	switch (index)
	{
	case 0:
		ui->rbtn_upLight->setChecked(state);
		break;
	case 1:
		ui->rbtn_downLight->setChecked(state);
		break;
	case 2:
		ui->rbtn_sideLight->setChecked(state);
		break;
	case 3:
		ui->rbtn_strobe->setChecked(state);
		break;
	default:
		break;
	}
}

void ButtonScanner::imgDis1_clicked()
{
	_dlgRealTimeImgDis->setGboxTitle("1号工位");
	_currentRealTimeDisIndex = 1;
	_dlgRealTimeImgDis->show();
}

void ButtonScanner::imgDis2_clicked()
{
	_dlgRealTimeImgDis->setGboxTitle("2号工位");
	_currentRealTimeDisIndex = 2;
	_dlgRealTimeImgDis->show();
}

void ButtonScanner::imgDis3_clicked()
{
	_dlgRealTimeImgDis->setGboxTitle("3号工位");
	_currentRealTimeDisIndex = 3;
	_dlgRealTimeImgDis->show();
}

void ButtonScanner::imgDis4_clicked()
{
	_dlgRealTimeImgDis->setGboxTitle("4号工位");
	_currentRealTimeDisIndex = 4;
	_dlgRealTimeImgDis->show();
}

void ButtonScanner::onCamera1Display(QPixmap image)
{
	if (!_isRealTimeDis)
	{
		imgDis1->setPixmap(image.scaled(imgDis1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else
	{
		if (_currentRealTimeDisIndex==1) {
			_dlgRealTimeImgDis->setShowImg(image);
		}
	}
}

void ButtonScanner::onCamera2Display(QPixmap image)
{
	if (!_isRealTimeDis)
	{
		imgDis2->setPixmap(image.scaled(imgDis2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else
	{
		if (_currentRealTimeDisIndex == 2) {
			_dlgRealTimeImgDis->setShowImg(image);
		}
	}
}

void ButtonScanner::onCamera3Display(QPixmap image)
{
	if (!_isRealTimeDis)
	{
		imgDis3->setPixmap(image.scaled(imgDis3->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else
	{
		if (_currentRealTimeDisIndex == 3) {
			_dlgRealTimeImgDis->setShowImg(image);
		}
	}
}

void ButtonScanner::onCamera4Display(QPixmap image)
{
	if (!_isRealTimeDis)
	{
		imgDis4->setPixmap(image.scaled(imgDis4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
	else
	{
		if (_currentRealTimeDisIndex == 4) {
			_dlgRealTimeImgDis->setShowImg(image);
		}
	}
}

void ButtonScanner::updateCameraLabelState(int cameraIndex, bool state)
{
	switch (cameraIndex)
	{
	case 1:
		if (state) {
			ui->label_camera1State->setText("连接成功");
			ui->label_camera1State->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
		}
		else {
			ui->label_camera1State->setText("连接失败");
			ui->label_camera1State->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
			rw::rqw::WarningInfo info;
			info.message = "相机1断连";
			info.type = rw::rqw::WarningType::Error;
			info.warningId = WarningId::ccameraDisconnectAlarm1;
			labelWarning->addWarning(info);
		}
		break;
	case 2:
		if (state) {
			ui->label_camera2State->setText("连接成功");
			ui->label_camera2State->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
		}
		else {
			ui->label_camera2State->setText("连接失败");
			ui->label_camera2State->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
			rw::rqw::WarningInfo info;
			info.message = "相机2断连";
			info.type = rw::rqw::WarningType::Error;
			info.warningId = WarningId::ccameraDisconnectAlarm2;
			labelWarning->addWarning(info);
		}
		break;
	case 3:
		if (state) {
			ui->label_camera3State->setText("连接成功");
			ui->label_camera3State->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
		}
		else {
			ui->label_camera3State->setText("连接失败");
			ui->label_camera3State->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
			rw::rqw::WarningInfo info;
			info.message = "相机3断连";
			info.type = rw::rqw::WarningType::Error;
			info.warningId = WarningId::ccameraDisconnectAlarm3;
			labelWarning->addWarning(info);
		}
		break;
	case 4:
		if (state) {
			ui->label_camera4State->setText("连接成功");
			ui->label_camera4State->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
		}
		else {
			ui->label_camera4State->setText("连接失败");
			ui->label_camera4State->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
			rw::rqw::WarningInfo info;
			info.message = "相机4断连";
			info.type = rw::rqw::WarningType::Error;
			info.warningId = WarningId::ccameraDisconnectAlarm4;
			labelWarning->addWarning(info);
		}
		break;
	default:
		break;
	}
}

void ButtonScanner::updateCardLabelState(bool state)
{
	isConnnectCard = state;
	if (state) {
		ui->label_cardState->setText("连接成功");
		ui->label_cardState->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
	}
	else {
		ui->label_cardState->setText("连接失败");
		ui->label_cardState->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
	}
}

void ButtonScanner::updateStatisticalInfoUI()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& mainWindowConfig = globalStruct.mainWindowConfig;
	auto productionYield = globalStruct.statisticalInfo.productionYield.load();
	auto produceCount = globalStruct.statisticalInfo.produceCount.load();
	auto wasteCount = globalStruct.statisticalInfo.wasteCount.load();
	auto removeRate = globalStruct.statisticalInfo.removeRate.load();
	mainWindowConfig.passRate = productionYield;
	mainWindowConfig.totalProduction = produceCount;
	mainWindowConfig.totalWaste = wasteCount;
	mainWindowConfig.scrappingRate = removeRate;
	ui->label_produceTotalValue->setText(QString::number(produceCount));
	ui->label_wasteProductsValue->setText(QString::number(wasteCount));
	ui->label_productionYieldValue->setText(QString::number(productionYield, 'f', 2) + QString(" %"));
	ui->label_removeRate->setText(QString::number(static_cast<int>(removeRate)) + QString(" /min"));
}

void ButtonScanner::pbtn_set_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		if (numKeyBord.getValue() == "1234") {
			_dlgProduceLineSet->setFixedSize(this->width(), this->height());
			_dlgProduceLineSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
			_dlgProduceLineSet->exec();
			ui->pbtn_beltSpeed->setText(QString::number(GlobalStructData::getInstance().dlgProduceLineSetConfig.motorSpeed));
		}
		else {
			QMessageBox::warning(this, "Error", "密码错误，请重新输入");
		}
	}
}

void ButtonScanner::pbtn_newProduction_clicked()
{
	auto& globalStrut = GlobalStructData::getInstance();
	auto currentRunningState = globalStrut.runningState.load();
	if (currentRunningState == RunningState::OpenRemoveFunc)
	{
		QMessageBox::warning(this, "错误", "请先停止生产线");
		return;
	}
	if (dlgNewProduction != nullptr) {
		currentRunningState = RunningState::Stop;
		ui->rbtn_debug->setChecked(false);
		rbtn_debug_checked(false);
		dlgNewProduction->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		dlgNewProduction->show();
	}
}

void ButtonScanner::pbtn_beltSpeed_clicked()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto currentRunningState = globalStruct.runningState.load();
	if (currentRunningState == RunningState::OpenRemoveFunc)
	{
		QMessageBox::warning(this, "错误", "请先停止生产线");
	}
	else {
		NumberKeyboard numKeyBord;
		numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		auto isAccept = numKeyBord.exec();
		if (isAccept == QDialog::Accepted) {
			auto value = numKeyBord.getValue();
			if (value.toDouble() < 0 || value.toDouble() > 1200)
			{
				QMessageBox::warning(this, "提示", "请输入0-1200之间的值");
				return;
			}
			auto& GlobalStructData = GlobalStructData::getInstance();
			ui->pbtn_beltSpeed->setText(value);
			GlobalStructData.mainWindowConfig.beltSpeed = value.toDouble();
			GlobalStructData.dlgProduceLineSetConfig.motorSpeed = ui->pbtn_beltSpeed->text().toDouble();
			_dlgProduceLineSet->updateBeltSpeed();
		}
	}
}

void ButtonScanner::pbtn_score_clicked()
{
	_dlgProductSet->readConfig();
	_dlgProductSet->setFixedSize(this->width(), this->height());
	_dlgProductSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgProductSet->exec();
}

void ButtonScanner::pbtn_resetProduct_clicked()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.mainWindowConfig.totalProduction = 0;
	globalStruct.mainWindowConfig.totalWaste = 0;
	globalStruct.mainWindowConfig.passRate = 0;
	globalStruct.mainWindowConfig.scrappingRate = 0;
	ui->label_produceTotalValue->setText(QString::number(globalStruct.mainWindowConfig.totalProduction));
	ui->label_wasteProductsValue->setText(QString::number(globalStruct.mainWindowConfig.totalWaste));
	ui->label_productionYieldValue->setText(QString::number(globalStruct.mainWindowConfig.passRate) + QString(" %"));
	ui->label_removeRate->setText(QString::number(globalStruct.mainWindowConfig.scrappingRate) + QString(" /min"));
	globalStruct.statisticalInfo.produceCount = 0;
	globalStruct.statisticalInfo.wasteCount = 0;
	globalStruct.statisticalInfo.productionYield = 0;
	globalStruct.statisticalInfo.removeRate = 0;
	globalStruct.saveConfig();
}

void ButtonScanner::pbtn_openSaveLocation_clicked()
{
	auto& globalStruct = GlobalStructData::getInstance();
	QString imageSavePath = globalStruct.imageSaveEngine->getRootPath();
	_picturesViewer->setRootPath(imageSavePath);
	_picturesViewer->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_picturesViewer->show();
}

void ButtonScanner::rbtn_debug_checked(bool checked)
{
	auto isRuning = ui->rbtn_removeFunc->isChecked();
	//if (!isRuning) {
	//	if (checked) {
	//		_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
	//		auto& GlobalStructData = GlobalStructData::getInstance();
	//		GlobalStructData.mainWindowConfig.isDebugMode = checked;
	//		GlobalStructData.runningState = RunningState::Debug;
	//	}
	//	else {
	//		auto& GlobalStructData = GlobalStructData::getInstance();
	//		GlobalStructData.mainWindowConfig.isDebugMode = checked;
	//		GlobalStructData.runningState = RunningState::Stop;
	//		_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
	//	}
	//}
	//else {
	//	ui->rbtn_debug->setChecked(false);
	//}
	auto& GlobalStructData = GlobalStructData::getInstance();
	auto& GlobalThread = GlobalStructThread::getInstance();
	if (!isRuning) {
		if (checked) {
			_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
			GlobalStructData.mainWindowConfig.isDebugMode = checked;
			GlobalStructData.runningState = RunningState::Debug;
			//GlobalThread.strobeLightThread->startThread();
			ui->rbtn_takePicture->setChecked(false);
			rbtn_takePicture_checked(false);
		}
		else {
			_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
			GlobalStructData.mainWindowConfig.isDebugMode = checked;
			GlobalStructData.runningState = RunningState::Stop;
			//GlobalThread.strobeLightThread->stopThread();
		}
		ui->cBox_isDisplayRec->setVisible(checked);
		ui->cBox_isDisplayText->setVisible(checked);
	}
	else {
		ui->rbtn_debug->setChecked(false);
	}
}

void ButtonScanner::rbtn_takePicture_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	if (GlobalStructData.runningState == RunningState::Debug)
	{
		ui->rbtn_takePicture->setChecked(false);
		return;
	}
	GlobalStructData.mainWindowConfig.isTakePictures = checked;
	GlobalStructData.saveConfig();
	GlobalStructData.isTakePictures = checked;
}

void ButtonScanner::rbtn_removeFunc_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.isEliminating = checked;
	GlobalStructData.saveConfig();
}

void ButtonScanner::rbtn_upLight_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.upLight = checked;
	GlobalStructData.saveConfig();
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motionPtr->SetIOOut(ControlLines::upLightOut, checked);
}

void ButtonScanner::rbtn_sideLight_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.sideLight = checked;
	GlobalStructData.saveConfig();
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motionPtr->SetIOOut(ControlLines::sideLightOut, checked);
}

void ButtonScanner::rbtn_downLight_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.downLight = checked;
	GlobalStructData.saveConfig();
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motionPtr->SetIOOut(ControlLines::downLightOut, checked);
}

void ButtonScanner::rbtn_defect_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.isDefect = checked;
	GlobalStructData.saveConfig();
	GlobalStructData.isOpenDefect = checked;
}

void ButtonScanner::rbtn_forAndAgainst_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.isPositive = checked;
	GlobalStructData.saveConfig();
}

void ButtonScanner::rbtn_strobe_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.strobeLight = checked;
	GlobalStructData.saveConfig();
	auto& motionPtr = zwy::scc::GlobalMotion::getInstance().motionPtr;
	motionPtr->SetIOOut(ControlLines::strobeLightOut, checked);
}

void ButtonScanner::cBox_isDisplayRec_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.debug_isDisplayRec = checked;
}

void ButtonScanner::cBox_isDisplayText_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.debug_isDisplayText = checked;
}

void ButtonScanner::labelClickable_title_clicked()
{
	auto& global = GlobalStructData::getInstance();
	if (global.isTrainModel)
	{
		QMessageBox::warning(this, "警告", "正在训练模型，请稍后再试");
		return;
	}
	auto currentRunningState = global.runningState.load();
	if (currentRunningState == RunningState::OpenRemoveFunc)
	{
		QMessageBox::warning(this, "警告", "正在运行剔废功能，请关闭后再试");
		return;
	}
	_dlgModelManager->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgModelManager->show();
}

void ButtonScanner::labelVersion_clicked()
{
	_dlgVersion->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgVersion->show();
}

void ButtonScanner::onAddWarningInfo(QString message, bool updateTimestampIfSame, int redDuration)
{
	rw::rqw::WarningInfo info;
	info.message = message;
	info.type = rw::rqw::WarningType::Error;
	labelWarning->addWarning(info, updateTimestampIfSame, redDuration);
	//updateCardLabelState(false);
}

void ButtonScanner::updateExposureTimeValueOnDlg(int exposureTime)
{
	_dlgExposureTimeSet->setExposureTime(exposureTime);
}

void ButtonScanner::checkPosiviveRadioButtonCheck()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.mainWindowConfig.isPositive = true;
	globalStruct.mainWindowConfig.isDefect = false;
	ui->rbtn_ForAndAgainst->setChecked(true);
	ui->rbtn_defect->setChecked(false);
}

void ButtonScanner::showDlgWarn(rw::rqw::WarningInfo info)
{
	QString timeStr = info.timestamp.toString("hh时mm分ss秒");
	dlgWarn->setTime(timeStr);
	dlgWarn->setText(info.message);
	switch (info.type)
	{
	case rw::rqw::WarningType::Error:
		dlgWarn->setTitle("错误");
		break;
	case rw::rqw::WarningType::Warning:
		dlgWarn->setTitle("警告");
		break;
	case rw::rqw::WarningType::Info:
		dlgWarn->setTitle("信息");
		break;
	default:
		dlgWarn->setTitle("未知");
		break;
	}

	dlgWarn->show();
}

void ButtonScanner::dlgWarningAccept()
{
	auto& thread = GlobalStructThread::getInstance().detachUtiltyThread;
	thread->isProcessing = false;
	thread->isProcessFinish = true;
}

void ButtonScanner::workTriggerError(int index)
{
	if (index == 1)
	{
		rw::rqw::WarningInfo info;
		info.message = "一工位运行中长时间无触发";
		info.type = rw::rqw::WarningType::Warning;
		info.warningId = WarningId::cworkTrigger1;
		labelWarning->addWarning(info);
	}
	if (index == 2)
	{
		rw::rqw::WarningInfo info;
		info.message = "二工位运行中长时间无触发";
		info.type = rw::rqw::WarningType::Warning;
		info.warningId = WarningId::cworkTrigger2;
		labelWarning->addWarning(info);
	}
	if (index == 3)
	{
		rw::rqw::WarningInfo info;
		info.message = "三工位运行中长时间无触发";
		info.type = rw::rqw::WarningType::Warning;
		info.warningId = WarningId::cworkTrigger3;
		labelWarning->addWarning(info);
	}
	if (index == 4)
	{
		rw::rqw::WarningInfo info;
		info.message = "四工位运行中长时间无触发";
		info.type = rw::rqw::WarningType::Warning;
		info.warningId = WarningId::cworkTrigger4;
		labelWarning->addWarning(info);
	}
}

void ButtonScanner::closeTakePictures()
{
	ui->rbtn_takePicture->setChecked(false);
	rbtn_takePicture_checked(false);
}

void ButtonScanner::shutdownComputerTrigger(int time)
{
	if (!isConnnectCard)
	{
		return;
	}

	int shutDownBoundary = 7;
	if (time == -1)
	{
		_dlgShutdownWarn->close();
		return;
	}
	if (time == 0)
	{
		_dlgShutdownWarn->show();
		_dlgShutdownWarn->setTimeValue(shutDownBoundary - time);
		return;
	}

	_dlgShutdownWarn->setTimeValue((shutDownBoundary - time) % shutDownBoundary);

	if ((shutDownBoundary - time) == 0)
	{
		isShutdownByIO = true;
		this->close();
	}
}

void ButtonScanner::pbtn_exit_clicked()
{
	isShutdownByIO = false;
	this->close();
}