#include "ZipperScanner.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"
#include "Utilty.hpp"
#include "WarnUtilty.hpp"

ZipperScanner::ZipperScanner(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::ZipperScannerClass())
{
	ui->setupUi(this);

	// 读取参数
	read_config();

	// 构建UI
	build_ui();

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


	// 连接相机1出图
	QObject::connect(GlobalStructDataZipper::getInstance().camera1.get(),
		&rw::rqw::CameraPassiveThread::frameCaptured,
		this, &ZipperScanner::onFrameCaptured,Qt::QueuedConnection);

	// 连接相机2出图
	QObject::connect(GlobalStructDataZipper::getInstance().camera2.get(),
		&rw::rqw::CameraPassiveThread::frameCaptured,
		this, &ZipperScanner::onFrameCaptured, Qt::QueuedConnection);
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
	//_dlgExposureTimeSet->ResetCamera(); //启动设置相机为默认状态
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

void ZipperScanner::destroyComponents()
{



	// 销毁相机
	auto& globalStructData = GlobalStructDataZipper::getInstance();
	QObject::disconnect(globalStructData.camera1.get(),
		&rw::rqw::CameraPassiveThread::frameCaptured,
		this, &ZipperScanner::onFrameCaptured);
	QObject::disconnect(globalStructData.camera2.get(),
		&rw::rqw::CameraPassiveThread::frameCaptured,
		this, &ZipperScanner::onFrameCaptured);
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

void ZipperScanner::onFrameCaptured(cv::Mat frame, size_t index)
{
	if (index == 1) {
		QPixmap pixmap = cvMatToQPixmap(frame);
		ui->label_imgDisplay_1->setPixmap(pixmap);
	}
	else if (index == 2) {
		QPixmap pixmap = cvMatToQPixmap(frame);
		ui->label_imgDisplay_2->setPixmap(pixmap);
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

