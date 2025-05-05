#include "stdafx.h"

#include "ButtonScanner.h"
#include"NumberKeyboard.h"

#include"LoadingDialog.h"
#include"PicturesViewer.h"
#include"DlgProductSet.h"
#include"DlgProduceLineSet.h"
#include"GlobalStruct.h"
#include"ButtonUtilty.h"

#include"rqw_CameraObjectThread.hpp"
#include"rqw_CameraObject.hpp"
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
	auto& globalStructData = GlobalStructData::getInstance();
	auto isRuning = ui->rbtn_removeFunc->isChecked();
	if (!isRuning) {
		auto & isDebug = globalStructData.isDebugMode;
		_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
		_dlgExposureTimeSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		_dlgExposureTimeSet->exec(); // 显示对话框
		_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
		if (isDebug)
		{
			rbtn_debug_checked(isDebug);
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

	// 获取 gBoix_ImageDisplay 的中心点
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
	label_lightBulb->setGeometry(newX, newY, bulbWidth, bulbHeight);

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

	//Set RadioButton ,make sure these can be checked at the same time
	read_image();
	set_radioButton();
	build_mainWindowData();
	build_dlgProduceLineSet();
	build_dlgProductSet();
	build_dlgExposureTimeSet();
	build_dlgNewProduction();
	build_picturesViewer();
	build_dlgModelManager();
	this->labelClickable_title = new rw::rqw::ClickableLabel(this);
	labelWarning = new rw::rqw::LabelWarning(this);
	ui->gBox_warningInfo->layout()->replaceWidget(ui->label_warningInfo, labelWarning);
	delete ui->label_warningInfo;

	labelClickable_title->setText(ui->label_title->text());
	labelClickable_title->setStyleSheet(ui->label_title->styleSheet());
	ui->hLayout_title->replaceWidget(ui->label_title, labelClickable_title);
	delete ui->label_title;
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
	motionPtr->SetIOOut(1, false);

	motionPtr->SetIOOut(7, false);
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
	_picturesViewer = new PicturesViewer(this);
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
		globalStruct.mainWindowConfig = rw::cdm::ButtonScannerMainWindow();
		globalStruct.saveMainWindowConfig();
		return;
	}
	else {
		globalStruct.ReadMainWindowConfig();
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
		globalStruct.dlgProduceLineSetConfig = rw::cdm::ButtonScannerProduceLineSet();
		globalStruct.saveDlgProduceLineSetConfig();
		return;
	}
	else {
		globalStruct.ReadDlgProduceLineSetConfig();
	}
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
		globalStruct.dlgProductSetConfig = rw::cdm::ButtonScannerDlgProductSet();
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
		globalStruct.dlgExposureTimeSetConfig = rw::cdm::ButtonScannerDlgExposureTimeSet();
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
		globalStruct.dlgHideScoreSetConfig = rw::cdm::DlgHideScoreSet();
		globalStruct.saveDlgHideScoreSetConfig();
		return;
	}
	else {
		globalStruct.ReadDlgHideScoreSetConfig();
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
		labelWarning->addWarning("相机1连接失败");
	}

	auto build2Result = globalStruct.buildCamera2();
	updateCameraLabelState(2, build2Result);
	if (!build2Result)
	{
		labelWarning->addWarning("相机2连接失败");
	}

	auto build3Result = globalStruct.buildCamera3();
	updateCameraLabelState(3, build3Result);
	if (!build3Result)
	{
		labelWarning->addWarning("相机3连接失败");
	}

	auto build4Result = globalStruct.buildCamera4();
	updateCameraLabelState(4, build4Result);
	if (!build4Result)
	{
		labelWarning->addWarning("相机4连接失败");
	}

	_dlgExposureTimeSet->ResetCamera(); //启动设置相机为默认状态
}

void ButtonScanner::build_imageProcessorModule()
{
	auto& globalStruct = GlobalStructData::getInstance();

	QDir dir;

	QString enginePathFull = globalPath.modelRootPath + globalPath.engineFileName;
	QString namePathFull = globalPath.modelRootPath + globalPath.nameFileName;
	QString onnxEnginePathOOFull = globalPath.modelRootPath + globalPath.onnxFileNameOO;
	QString onnxEnginePathSOFull = globalPath.modelRootPath + globalPath.onnxFileNameSO;

	QFileInfo engineFile(enginePathFull);
	QFileInfo nameFile(namePathFull);
	QFileInfo onnxEngineOOFile(onnxEnginePathOOFull);
	QFileInfo onnxEngineSOFile(onnxEnginePathSOFull);

	if (!engineFile.exists() || !nameFile.exists() || !onnxEngineOOFile.exists() || !onnxEngineSOFile.exists()) {
		QMessageBox::critical(this, "Error", "Engine file or Name file does not exist. The application will now exit.");
		QApplication::quit();
		return;
	}

	globalStruct.enginePath = enginePathFull;
	globalStruct.namePath = namePathFull;
	globalStruct.onnxEngineOOPath = onnxEnginePathOOFull;
	globalStruct.onnxEngineSOPath = onnxEnginePathSOFull;

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
		labelWarning->addWarning("运动控制器连接失败");
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
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(5, 5, true, tifeishijian1);
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
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(1, 4, true, tifeishijian2);
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
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(2, 3, true, tifeishijian3);
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
					zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(3, 2, true, tifeishijian4);
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
			state = motionPtr->GetIOIn(2);
			//急停

			if (state == true)
			{
				globalStruct.isOpenRemoveFunc = false;
				globalStruct.isDebugMode = false;
				QMetaObject::invokeMethod(qApp, [this, state]
					{
						ui->rbtn_removeFunc->setChecked(false);
						ui->rbtn_debug->setChecked(false);
						label_lightBulb->setVisible(true);
					});
					// pidaimove->stop();
					motionPtr->StopAllAxis();
					motionPtr->SetIOOut(1, false);

					motionPtr->SetIOOut(7, false);
			}
			else
			{
				//开始按钮
				bool state = false;
				state = motionPtr->GetIOIn(1);
				//启动程序
				if (state == true)
				{
					if (dlgNewProduction->_info.isActivate == false)
					{
						globalStruct.isOpenRemoveFunc = true;
						globalStruct.isDebugMode = false;
						QMetaObject::invokeMethod(qApp, [this, state]
							{
								_dlgExposureTimeSet->ResetCamera();
								ui->rbtn_removeFunc->setChecked(true);
								ui->rbtn_debug->setChecked(false);
								label_lightBulb->setVisible(false);
							});
					}
					//所有电机上电
					QtConcurrent::run([this, &motionPtr]() {
						QThread::msleep(500);
						motionPtr->SetIOOut(1, true);
						//启动电机
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
						});
				}
				//停止点
				state = motionPtr->GetIOIn(2);
				if (state)
				{
					globalStruct.isOpenRemoveFunc = false;
					globalStruct.isDebugMode = false;
					QMetaObject::invokeMethod(qApp, [this, state]
						{
							ui->rbtn_removeFunc->setChecked(false);
							ui->rbtn_debug->setChecked(false);
							label_lightBulb->setVisible(false);
						});
						motionPtr->StopAllAxis();
						motionPtr->SetIOOut(1, false);
						motionPtr->SetIOOut(7, false);
				}

				//获取气压表数据
				auto qiya = motionPtr->GetIOIn(7);
				if (qiya == true) {
					//气压正常
					motionPtr->SetIOOut(8, true);
					QMetaObject::invokeMethod(qApp, [this, state]
						{
							labelWarning->addWarning("气压不正常", true);
						});
				}
				else {

						motionPtr->SetIOOut(8, false);
				}

				if (globalStruct.mainWindowConfig.upLight) {
					motionPtr->SetIOOut(9, true);
				}
				else {
					motionPtr->SetIOOut(9, false);
				}

				if (globalStruct.mainWindowConfig.downLight) {
					motionPtr->SetIOOut(10, true);
				}
				else {
					motionPtr->SetIOOut(10, false);
				}

				if (globalStruct.mainWindowConfig.sideLight) {
					motionPtr->SetIOOut(0, true);
				}
				else {
					motionPtr->SetIOOut(0, false);
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
	QObject::connect(globalStruct.statisticalInfoComputingThread.get(), &StatisticalInfoComputingThread::updateStatisticalInfo,
		this, &ButtonScanner::updateStatisticalInfoUI, Qt::QueuedConnection);
	QObject::connect(globalStruct.statisticalInfoComputingThread.get(), &StatisticalInfoComputingThread::addWarningInfo,
		this, &ButtonScanner::onAddWarningInfo, Qt::QueuedConnection);

	QObject::connect(globalStruct.monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::updateCameraLabelState,
		this, &ButtonScanner::updateCameraLabelState, Qt::QueuedConnection);
	QObject::connect(globalStruct.monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::updateCardLabelState,
		this, &ButtonScanner::updateCardLabelState, Qt::QueuedConnection);
	QObject::connect(globalStruct.monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::addWarningInfo,
		this, &ButtonScanner::onAddWarningInfo, Qt::QueuedConnection);
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
	label_lightBulb->setGeometry(newX+30, newY, bulbWidth, bulbHeight);
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
	default:
		break;
	}
}

void ButtonScanner::onCamera1Display(QPixmap image)
{
	ui->label_imgDisplay->setPixmap(image.scaled(ui->label_imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ButtonScanner::onCamera2Display(QPixmap image)
{
	ui->label_imgDisplay_2->setPixmap(image.scaled(ui->label_imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ButtonScanner::onCamera3Display(QPixmap image)
{
	ui->label_imgDisplay_3->setPixmap(image.scaled(ui->label_imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ButtonScanner::onCamera4Display(QPixmap image)
{
	ui->label_imgDisplay_4->setPixmap(image.scaled(ui->label_imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
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
		}
		break;
	default:
		break;
	}
}

void ButtonScanner::updateCardLabelState(bool state)
{
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
	if (globalStrut.isOpenRemoveFunc)
	{
		QMessageBox::warning(this, "错误", "请先停止生产线");
		return;
	}
	if (dlgNewProduction != nullptr) {
		dlgNewProduction->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		dlgNewProduction->show();
	}
}

void ButtonScanner::pbtn_beltSpeed_clicked()
{
	auto& globalStruct = GlobalStructData::getInstance();
	if (globalStruct.isOpenRemoveFunc)
	{
		QMessageBox::warning(this, "错误", "请先停止生产线");
	}
	else {
		NumberKeyboard numKeyBord;
		numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		auto isAccept = numKeyBord.exec();
		if (isAccept == QDialog::Accepted) {
			auto value = numKeyBord.getValue();
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
	if (!isRuning) {
		if (checked) {
			_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
			auto& GlobalStructData = GlobalStructData::getInstance();
			GlobalStructData.mainWindowConfig.isDebugMode = checked;
			GlobalStructData.isDebugMode = checked;
		}
		else {
			auto& GlobalStructData = GlobalStructData::getInstance();
			GlobalStructData.mainWindowConfig.isDebugMode = checked;
			GlobalStructData.isDebugMode = checked;
			_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
		}
	}
	else {
		ui->rbtn_debug->setChecked(false);
	}
}

void ButtonScanner::rbtn_takePicture_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
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
}

void ButtonScanner::rbtn_sideLight_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.sideLight = checked;
	GlobalStructData.saveConfig();
}

void ButtonScanner::rbtn_downLight_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.mainWindowConfig.downLight = checked;
	GlobalStructData.saveConfig();
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

void ButtonScanner::labelClickable_title_clicked()
{
	auto& global = GlobalStructData::getInstance();
	if (global.isTrainModel)
	{
		QMessageBox::warning(this, "警告", "正在训练模型，请稍后再试");
		return;
	}
	if (global.isOpenRemoveFunc)
	{
		QMessageBox::warning(this, "警告", "正在运行剔废功能，请关闭后再试");
		return;
	}
	_dlgModelManager->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgModelManager->show();
}

void ButtonScanner::onAddWarningInfo(QString message, bool updateTimestampIfSame, int redDuration)
{
	labelWarning->addWarning(message, updateTimestampIfSame, redDuration);
}

void ButtonScanner::pbtn_exit_clicked()
{
	this->close();
}