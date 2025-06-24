#include "ZipperScanner.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"
#include "Utilty.hpp"
#include "WarnUtilty.hpp"
#include "rqw_CameraObjectThread.hpp"
#include "DetachDefectThread.h"

ZipperScanner::ZipperScanner(QWidget* parent)
	: QMainWindow(parent)
	, ui(new Ui::ZipperScannerClass())
{
	ui->setupUi(this);

	// 读取参数
	read_config();

	// 构建UI
	build_ui();


	auto& globalStruct = GlobalStructDataZipper::getInstance();

	globalStruct.build_motion();
	// 构建优先队列
	globalStruct.build_PriorityQueue();



	// 构建异步剔废线程
	globalStruct.build_DetachDefectThreadZipper();

	// 构建相机与板卡重连线程
	globalStruct.build_CameraAndCardStateThreadZipper();

	// 构建图像保存引擎
	build_imageSaveEngine();

	// 构建图像处理模块
	build_imageProcessorModule();

	// 连接相机
	build_camera();

	// 连接槽函数
	build_connect();

	// 启用所有后台线程
	build_threads();
}

ZipperScanner::~ZipperScanner()
{
	destroyComponents();
	delete ui;
}

// 构建UI
void ZipperScanner::build_ui()
{
	build_ZipperScannerData();
	build_DlgProductSetData();
	build_DlgProductScore();
	build_DlgExposureTimeSet();

}

// 连接槽函数
void ZipperScanner::build_connect()
{
	auto& GlobalStructDataZipper = GlobalStructDataZipper::getInstance();
	// 退出
	QObject::connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &ZipperScanner::pbtn_exit_clicked);

	// 设置
	QObject::connect(ui->pbtn_set, &QPushButton::clicked,
		this, &ZipperScanner::pbtn_set_clicked);

	// 分数
	QObject::connect(ui->pbtn_score, &QPushButton::clicked,
		this, &ZipperScanner::pbtn_score_clicked);

	// 开启调试显示新窗体
	QObject::connect(ui->rbtn_debug, &QRadioButton::clicked,
		this, &ZipperScanner::rbtn_debug_checked);

	// 强光
	QObject::connect(ui->rbtn_strongLight, &QRadioButton::clicked,
		this, &ZipperScanner::rbtn_strongLight_checked);
	// 中光
	QObject::connect(ui->rbtn_mediumLight, &QRadioButton::clicked,
		this, &ZipperScanner::rbtn_mediumLight_checked);
	// 弱光
	QObject::connect(ui->rbtn_weakLight, &QRadioButton::clicked,
		this, &ZipperScanner::rbtn_weakLight_checked);

	// 打开图像路径文件夹
	QObject::connect(ui->pbtn_openSaveLocation, &QPushButton::clicked,
		this, &ZipperScanner::pbtn_openSaveLocation_clicked);

	// 采图
	QObject::connect(ui->rbtn_takePicture, &QRadioButton::clicked,
		this, &ZipperScanner::rbtn_takePicture_checked);

	// 剔废功能
	QObject::connect(ui->rbtn_removeFunc, &QRadioButton::clicked,
		this, &ZipperScanner::rbtn_removeFunc_checked);

	// 是否识别框
	QObject::connect(ui->ckb_shibiekuang, &QCheckBox::clicked,
		this, &ZipperScanner::ckb_shibiekuang_checked);

	// 是否文字
	QObject::connect(ui->ckb_wenzi, &QCheckBox::clicked,
		this, &ZipperScanner::ckb_wenzi_checked);

	// 连接显示NG图像
	QObject::connect(GlobalStructDataZipper.modelCamera1.get(), &ImageProcessingModuleZipper::imageNGReady,
		this, &ZipperScanner::onCameraNGDisplay);
	QObject::connect(GlobalStructDataZipper.modelCamera2.get(), &ImageProcessingModuleZipper::imageNGReady,
		this, &ZipperScanner::onCameraNGDisplay);

	// 连接UI更新
	QObject::connect(&GlobalStructDataZipper.getInstance(), &GlobalStructDataZipper::emit_updateUiLabels,
		this,&ZipperScanner::updateUiLabels);

}

// 构建相机
void ZipperScanner::build_camera()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	globalStruct.cameraIp1 = "11";
	globalStruct.cameraIp2 = "12";

	auto build1Result = globalStruct.buildCamera1();
	updateCameraLabelState(1, build1Result);
	if (!build1Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机1连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm1;
		info.type = rw::rqw::WarningType::Error;
		//label_warningInfo->addWarning(info);
	}
	auto build2Result = false;
	if (globalStruct.setConfig.qiyongerxiangji)
	{
		build2Result = globalStruct.buildCamera2();
	}
	updateCameraLabelState(2, build2Result);
	if (!build2Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机2连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm2;
		info.type = rw::rqw::WarningType::Error;
		//label_warningInfo->addWarning(info);
	}
}

// 加载ZipperScanner窗体数据
void ZipperScanner::build_ZipperScannerData()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& zipperScannerConfig = globalStruct.generalConfig;
	// 初始化全局数据
	ui->label_produceTotalValue->setText(QString::number(zipperScannerConfig.totalProductionVolume));
	ui->label_wasteProductsValue->setText(QString::number(zipperScannerConfig.totalDefectiveVolume));
	ui->label_productionYieldValue->setText(QString::number(zipperScannerConfig.productionYield) + QString(" %"));
	ui->rbtn_strongLight->setChecked(zipperScannerConfig.qiangGuang);
	ui->rbtn_mediumLight->setChecked(zipperScannerConfig.zhongGuang);
	ui->rbtn_weakLight->setChecked(zipperScannerConfig.ruoGuang);

	// 去掉标题栏
	this->setWindowFlags(Qt::FramelessWindowHint);

	// 开机默认不显示但是勾选
	ui->ckb_shibiekuang->setVisible(false);
	ui->ckb_wenzi->setVisible(false);

	ui->ckb_shibiekuang->setChecked(true);
	ui->ckb_wenzi->setChecked(true);

	globalStruct.buildImageSaveEngine();

	// 初始化图像查看器
	_picturesViewer = new PictureViewerThumbnails(this);
}

// 通过实现DlgProductSet的构造函数进行初始化
void ZipperScanner::build_DlgProductSetData()
{
	_dlgProductSet = new DlgProductSet(this);
}

// 通过实现DlgProductScore的构造函数进行初始化
void ZipperScanner::build_DlgProductScore()
{
	_dlgProductScore = new DlgProductScore(this);
}

void ZipperScanner::build_DlgExposureTimeSet()
{
	_dlgExposureTimeSet = new DlgExposureTimeSet(this);
}

void ZipperScanner::build_imageProcessorModule()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	QDir dir;

	QString enginePathFull = globalPath.modelPath;

	QFileInfo engineFile(enginePathFull);

	if (!engineFile.exists()) {
		QMessageBox::critical(this, "Error", "Engine file or Name file does not exist. The application will now exit.");
		QApplication::quit();
		return;
	}

	globalStruct.buildImageProcessorModules(enginePathFull);

	QObject::connect(globalStruct.modelCamera1.get(), &ImageProcessingModuleZipper::imageReady, this, &ZipperScanner::onCamera1Display);
	QObject::connect(globalStruct.modelCamera2.get(), &ImageProcessingModuleZipper::imageReady, this, &ZipperScanner::onCamera2Display);

}

void ZipperScanner::build_imageSaveEngine()
{
	QDir dir;
	QString imageSavePath = globalPath.imageSaveRootPath;
	//清理旧的数据

	//获取当前日期并设置保存路径
	QString currentDate = QDate::currentDate().toString("yyyy_MM_dd");
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	globalStruct.buildImageSaveEngine();
	QString imageSaveEnginePath = imageSavePath + currentDate;

	QString imagesFilePathFilePathFull = dir.absoluteFilePath(imageSaveEnginePath);
	globalStruct.imageSaveEngine->setRootPath(imagesFilePathFilePathFull);
	globalStruct.imageSaveEngine->startEngine();
}

void ZipperScanner::build_threads()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	// 启动异步剔废线程
	globalStruct.detachDefectThreadZipper->startThread();
	// 启动相机重连线程
	globalStruct.cameraAndCardStateThreadZipper->startThread();
}

void ZipperScanner::destroyComponents()
{

	auto& globalStructData = GlobalStructDataZipper::getInstance();
	globalStructData.destory_motion();
	// 销毁相机
	globalStructData.destroyCamera();
	// 销毁图像处理模块
	globalStructData.destroyImageProcessingModule();
	// 销毁图像保存模块
	globalStructData.destroyImageSaveEngine();
	// 销毁异步剔废线程
	globalStructData.destroy_DetachDefectThreadZipper();
	// 销毁剔废优先队列
	globalStructData.destroy_PriorityQueue();
	// 保存参数
	globalStructData.saveGeneralConfig();
}

void ZipperScanner::read_config()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	globalStruct.buildConfigManager(rw::oso::StorageType::Xml);

	read_config_GeneralConfig();
	read_config_ScoreConfig();
	read_config_SetConfig();

}

// 读取通用配置
void ZipperScanner::read_config_GeneralConfig()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	auto& generalConfigPath = globalPath.generalConfigPath;

	QFileInfo generalConfigFile(generalConfigPath);

	// 如果文件不存在
	if (!generalConfigFile.exists())
	{
		QDir configDir = QFileInfo(generalConfigPath).absoluteDir();
		if (!configDir.exists())
		{
			configDir.mkpath(".");
		}
		QFile file(generalConfigPath);
		if (file.open(QIODevice::WriteOnly))
		{
			file.close();
		}
		else
		{
			QMessageBox::critical(this, "Error", "无法创建配置文件generalConfig.xml");
		}
		globalStruct.generalConfig = cdm::GeneralConfig();
		globalStruct.storeContext->save(globalStruct.generalConfig, globalPath.generalConfigPath.toStdString());
		return;
	}
	else
	{
		globalStruct.generalConfig = *globalStruct.storeContext->load(globalPath.generalConfigPath.toStdString());
		// 加载主窗体UI的设置
		globalStruct.generalConfig.isDefect = true; // 默认开启剔废模式
		rbtn_stop_clicked(true); // 默认停止
		rbtn_removeFunc_checked(true);
		ui->rbtn_removeFunc->setChecked(globalStruct.generalConfig.isDefect);
		ui->rbtn_strongLight->setChecked(globalStruct.generalConfig.qiangGuang);
		ui->rbtn_mediumLight->setChecked(globalStruct.generalConfig.zhongGuang);
		ui->rbtn_weakLight->setChecked(globalStruct.generalConfig.ruoGuang);
	}
}

// 读取分数配置
void ZipperScanner::read_config_ScoreConfig()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	auto& scoreConfigPathFull = globalPath.scoreConfigPath;

	QFileInfo scoreConfigFile(scoreConfigPathFull);

	if (!scoreConfigFile.exists())
	{
		QDir configDir = QFileInfo(scoreConfigPathFull).absoluteDir();
		if (!configDir.exists())
		{
			configDir.mkpath(".");
		}
		QFile file(scoreConfigPathFull);
		if (file.open(QIODevice::WriteOnly))
		{
			file.close();
		}
		else
		{
			QMessageBox::critical(this, "Error", "无法创建配置文件scoreConfig.xml");
		}
		globalStruct.scoreConfig = cdm::ScoreConfig();
		globalStruct.storeContext->save(globalStruct.scoreConfig, globalPath.scoreConfigPath.toStdString());
		return;
	}
	else
	{
		globalStruct.scoreConfig = *globalStruct.storeContext->load(globalPath.scoreConfigPath.toStdString());
	}
}

// 读取设置配置
void ZipperScanner::read_config_SetConfig()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& setConfigPathFull = globalPath.setConfigPath;
	QFileInfo setConfigFile(setConfigPathFull);
	if (!setConfigFile.exists())
	{
		QDir configDir = QFileInfo(setConfigPathFull).absoluteDir();
		if (!configDir.exists())
		{
			configDir.mkpath(".");
		}
		QFile file(setConfigPathFull);
		if (file.open(QIODevice::WriteOnly))
		{
			file.close();
		}
		else
		{
			QMessageBox::critical(this, "Error", "无法创建配置文件setConfig.xml");
		}
		globalStruct.setConfig = cdm::SetConfig();
		globalStruct.storeContext->save(globalStruct.setConfig, globalPath.setConfigPath.toStdString());
		return;
	}
	else
	{
		globalStruct.setConfig = *globalStruct.storeContext->load(globalPath.setConfigPath.toStdString());
	}
}

void ZipperScanner::pbtn_exit_clicked()
{
	this->close();
}

void ZipperScanner::pbtn_set_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		if (numKeyBord.getValue() == "1234") {
			_dlgProductSet->setFixedSize(this->width(), this->height());
			_dlgProductSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
			_dlgProductSet->exec();
		}
		else if (numKeyBord.getValue() == "6666")
		{
			_dlgExposureTimeSet->setFixedSize(500,300);
			_dlgExposureTimeSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
			_dlgExposureTimeSet->exec();
		}
		else {
			QMessageBox::warning(this, "Error", "密码错误，请重新输入");
		}
	}
}

void ZipperScanner::pbtn_score_clicked()
{
	_dlgProductScore->setFixedSize(this->width(), this->height());
	_dlgProductScore->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgProductScore->exec();
}

void ZipperScanner::rbtn_debug_checked(bool checked)
{
	auto isRuning = ui->rbtn_removeFunc->isChecked();

	auto& GlobalStructData = GlobalStructDataZipper::getInstance();
	if (!isRuning) {
		if (checked) {
			_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
			//GlobalStructData.generalConfig.isDebug = checked;
			GlobalStructData.runningState = RunningState::Debug;
			if (GlobalStructData.camera1)
			{
				GlobalStructData.camera1->setFrameRate(5);
			}
			if (GlobalStructData.camera2)
			{
				GlobalStructData.camera2->setFrameRate(5);
			}
			//GlobalThread.strobeLightThread->startThread();
			ui->rbtn_takePicture->setChecked(false);
			//rbtn_takePicture_checked(false);
		}
		else {
			_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
			//GlobalStructData.generalConfig.isDebug = checked;
			GlobalStructData.runningState = RunningState::Stop;
			//GlobalThread.strobeLightThread->stopThread();
		}
		ui->ckb_shibiekuang->setVisible(checked);
		ui->ckb_wenzi->setVisible(checked);
	}
	else {
		ui->rbtn_debug->setChecked(false);
	}
}

void ZipperScanner::rbtn_strongLight_checked(bool checked)
{
	auto& generalConfig = GlobalStructDataZipper::getInstance().generalConfig;
	if (checked)
	{
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		globalStruct.setLightLevel(LightLevel::StrongLight);
		generalConfig.zhongGuang = false;
		generalConfig.ruoGuang = false;
	}
	generalConfig.qiangGuang = ui->rbtn_strongLight->isChecked();
}

void ZipperScanner::rbtn_mediumLight_checked(bool checked)
{
	auto& generalConfig = GlobalStructDataZipper::getInstance().generalConfig;
	if (checked)
	{
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		globalStruct.setLightLevel(LightLevel::MediumLight);
		generalConfig.qiangGuang = false;
		generalConfig.ruoGuang = false;
	}
	generalConfig.zhongGuang = ui->rbtn_mediumLight->isChecked();
}

void ZipperScanner::rbtn_weakLight_checked(bool checked)
{
	auto& generalConfig = GlobalStructDataZipper::getInstance().generalConfig;
	if (checked)
	{
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		globalStruct.setLightLevel(LightLevel::WeakLight);
		generalConfig.qiangGuang = false;
		generalConfig.zhongGuang = false;
	}
	generalConfig.ruoGuang = ui->rbtn_weakLight->isChecked();
}

void ZipperScanner::pbtn_openSaveLocation_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	QString imageSavePath = globalStruct.imageSaveEngine->getRootPath();

	_picturesViewer->setRootPath(imageSavePath);
	_picturesViewer->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_picturesViewer->show();
}

void ZipperScanner::rbtn_takePicture_checked()
{
	if (ui->rbtn_debug->isChecked() == true)
	{
		ui->rbtn_takePicture->setChecked(false);
	}
	auto& generalConfig = GlobalStructDataZipper::getInstance().generalConfig;
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	generalConfig.isSaveImg = ui->rbtn_takePicture->isChecked();
	globalStruct.isTakePictures = ui->rbtn_takePicture->isChecked();
}

void ZipperScanner::rbtn_removeFunc_checked(bool checked)
{
	if (checked)
	{
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		globalStruct.runningState = RunningState::OpenRemoveFunc;
		_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
		if (globalStruct.camera1)
		{
			globalStruct.camera1->setFrameRate(50);
		}
		if (globalStruct.camera2)
		{
			globalStruct.camera2->setFrameRate(50);
		}
		ui->rbtn_debug->setChecked(false);
		ui->ckb_shibiekuang->setVisible(false);
		ui->ckb_wenzi->setVisible(false);
	}
	else
	{
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		globalStruct.runningState = RunningState::Stop;
	}
}

void ZipperScanner::ckb_shibiekuang_checked(bool checked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	globalStruct.generalConfig.isshibiekuang = ui->ckb_shibiekuang->isChecked();
}

void ZipperScanner::ckb_wenzi_checked(bool checked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	globalStruct.generalConfig.iswenzi = ui->ckb_wenzi->isChecked();
}

void ZipperScanner::rbtn_start_clicked(bool checked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	if (checked)
	{
		globalStruct.generalConfig.isStart = true;
		globalStruct.generalConfig.isStop = false;
	}
	else
	{
		globalStruct.generalConfig.isStart = false;
		globalStruct.generalConfig.isStop = true;
	}
}

void ZipperScanner::rbtn_stop_clicked(bool checked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	if (checked)
	{
		globalStruct.generalConfig.isStart = false;
		globalStruct.generalConfig.isStop = true;
	}
	else
	{
		globalStruct.generalConfig.isStart = true;
		globalStruct.generalConfig.isStop = false;
	}
}

void ZipperScanner::onCamera1Display(QPixmap image)
{
	ui->label_imgDisplay_1->setPixmap(image.scaled(ui->label_imgDisplay_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ZipperScanner::onCamera2Display(QPixmap image)
{
	ui->label_imgDisplay_3->setPixmap(image.scaled(ui->label_imgDisplay_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ZipperScanner::onCameraNGDisplay(QPixmap image, size_t index, bool isbad)
{
	if (isbad)
	{
		if (index == 1)
		{
			ui->label_imgDisplay_2->setPixmap(image.scaled(ui->label_imgDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		}
		else if (index == 2)
		{
			ui->label_imgDisplay_4->setPixmap(image.scaled(ui->label_imgDisplay_4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		}
	}
}

void ZipperScanner::updateUiLabels(int index, bool isConnected)
{
	switch (index)
	{
	case 1:
		if (isConnected) {
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
			//labelWarning->addWarning(info);
		}
		break;
	case 2:
		if (isConnected) {
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
			//labelWarning->addWarning(info);
		}
		break;
	default:
		break;
	}
}

void ZipperScanner::updateCameraLabelState(int cameraIndex, bool state)
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
			//labelWarning->addWarning(info);
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
			//labelWarning->addWarning(info);
		}
		break;
	default:
		break;
	}
}

