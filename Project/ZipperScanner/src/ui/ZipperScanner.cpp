#include "ZipperScanner.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"
#include "Utilty.hpp"
#include "WarnUtilty.hpp"
#include "rqw_CameraObjectThread.hpp"

ZipperScanner::ZipperScanner(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::ZipperScannerClass())
{
	ui->setupUi(this);

	// 读取参数
	read_config();

	// 构建UI
	build_ui();

	// 构建图像处理模块
	build_imageProcessorModule();

	// 连接相机
	build_camera();

	// 连接槽函数
	build_connect();
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

}

// 连接槽函数
void ZipperScanner::build_connect()
{
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
	else
	{
		QObject::connect(globalStruct.camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured, globalStruct.modelCamera1.get(), &ImageProcessingModuleZipper::onFrameCaptured);
	}
	auto build2Result = globalStruct.buildCamera2();
	updateCameraLabelState(2, build2Result);
	if (!build2Result)
	{
		rw::rqw::WarningInfo info;
		info.message = "相机2连接失败";
		info.warningId = WarningId::ccameraDisconnectAlarm2;
		info.type = rw::rqw::WarningType::Error;
		//label_warningInfo->addWarning(info);
	}
	else
	{
		QObject::connect(globalStruct.camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured, globalStruct.modelCamera2.get(), &ImageProcessingModuleZipper::onFrameCaptured);
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


	// 开机默认不显示但是勾选
	ui->ckb_shibiekuang->setVisible(false);
	ui->ckb_wenzi->setVisible(false);

	ui->ckb_shibiekuang->setChecked(true);
	ui->ckb_wenzi->setChecked(true);
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

void ZipperScanner::destroyComponents()
{



	// 销毁相机
	auto& globalStructData = GlobalStructDataZipper::getInstance();
	globalStructData.destroyCamera();
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

	auto &generalConfigPath = globalPath.generalConfigPath;

	QFileInfo generalConfigFile(generalConfigPath);

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
			//_dlgExposureTimeSet->SetCamera(); // 设置相机为实时采集
			//GlobalStructData.generalConfig.isDebug = checked;
			GlobalStructData.runningState = RunningState::Debug;
			//GlobalThread.strobeLightThread->startThread();
			ui->rbtn_takePicture->setChecked(false);
			//rbtn_takePicture_checked(false);
		}
		else {
			//_dlgExposureTimeSet->ResetCamera(); // 重置相机为硬件触发
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

void ZipperScanner::onCamera1Display(QPixmap image)
{
	ui->label_imgDisplay_1->setPixmap(image.scaled(ui->label_imgDisplay_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ZipperScanner::onCamera2Display(QPixmap image)
{
	ui->label_imgDisplay_2->setPixmap(image.scaled(ui->label_imgDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
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

