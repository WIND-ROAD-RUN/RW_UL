#include "SmartCroppingOfBags.h"

#include "oso_StorageContext.hpp"
#include "ui_SmartCroppingOfBags.h"
#include <GlobalStruct.hpp>
#include <QMessageBox>

#include "NumberKeyboard.h"

#include"ImageCollage.hpp"
#include "WarnUtilty.hpp"

SmartCroppingOfBags::SmartCroppingOfBags(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::SmartCroppingOfBagsClass())
{
	ui->setupUi(this);

	// 读取参数
	read_config();

	// 构建UI
	build_ui();

	//构建运动控制器
	build_motion();

	// 构建优先队列
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	globalStruct.build_PriorityQueue();

	// 构建异步剔废线程
	globalStruct.build_DetachDefectThreadSmartCroppingOfBags();
	globalStruct.build_MonitorIOSmartCroppingOfBags();

	// 构建图像保存引擎
	build_imageSaveEngine();

	// 构建图像处理模块
	build_imageProcessorModule();

	// 连接相机
	build_camera();

	// 连接槽函数
	build_connect();
}

SmartCroppingOfBags::~SmartCroppingOfBags()
{
	destroyComponents();
	delete ui;
}

void SmartCroppingOfBags::build_ui()
{
	build_SmartCroppingOfBagsData();
	build_DlgProductSetData();
	build_DlgProductScore();
}

void SmartCroppingOfBags::build_connect()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	QObject::connect(ui->btn_pingbiquyu, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_pingbiquyu_clicked);
	QObject::connect(ui->btn_chanliangqingling, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_chanliangqingling_clicked);
	QObject::connect(ui->btn_daizizhonglei, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_daizizhonglei_clicked);
	QObject::connect(ui->btn_down, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_down_clicked);
	QObject::connect(ui->btn_baoguang, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_baoguang_clicked);
	QObject::connect(ui->btn_up, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_up_clicked);
	QObject::connect(ui->btn_normalParam, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_normalParam_clicked);
	QObject::connect(ui->btn_setParam, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_setParam_clicked);
	QObject::connect(ui->rbtn_removeFunc, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_tifei_checked);
	QObject::connect(ui->ckb_debug, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_Debug_checked);
	QObject::connect(ui->ckb_cuntu, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_cuntu_checked);
	QObject::connect(ui->btn_close, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_close_clicked);
	QObject::connect(ui->rbtn_yinshuazhiliangjiance, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::rbtn_yinshuazhiliangjiance_clicked);
	QObject::connect(ui->rbtn_zhinengcaiqie, &QPushButton::clicked,
		this, &SmartCroppingOfBags::rbtn_zhinengcaiqie_clicked);

	// 连接显示NG图像
	QObject::connect(globalStruct.modelCamera1.get(), &ImageProcessingModuleSmartCroppingOfBags::imageNGReady,
		this, &SmartCroppingOfBags::onCameraNGDisplay);
	QObject::connect(globalStruct.modelCamera2.get(), &ImageProcessingModuleSmartCroppingOfBags::imageNGReady,
		this, &SmartCroppingOfBags::onCameraNGDisplay);

}

void SmartCroppingOfBags::build_motion()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto buildResult=globalStruct.build_motion();

	updateCardLabelState(buildResult);

}

void SmartCroppingOfBags::destroy_motion()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	globalStruct.destroy_motion();
}

void SmartCroppingOfBags::build_camera()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
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
	if (globalStruct.setConfig.qiyonger)
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

void SmartCroppingOfBags::build_SmartCroppingOfBagsData()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& smartCroppingOfBagsConfig = globalStruct.generalConfig;

	ui->lb_shengchanzongliang->setText(QString::number(globalStruct.generalConfig.shengchanzongliang));
	ui->lb_lianglv->setText(QString::number(globalStruct.generalConfig.shengchanlianglv));
	ui->lb_feipinshuliang->setText(QString::number(globalStruct.generalConfig.feipinshuliang));
	ui->lb_pingjundaichang->setText(QString::number(globalStruct.generalConfig.pingjundaichang));
	ui->rbtn_removeFunc->setChecked(globalStruct.generalConfig.istifei);
	ui->ckb_debug->setChecked(globalStruct.generalConfig.isDebug);
	ui->ckb_cuntu->setChecked(globalStruct.generalConfig.iscuntu);
	ui->btn_baoguang->setText(QString::number(globalStruct.generalConfig.baoguang));
	ui->rbtn_yinshuazhiliangjiance->setChecked(globalStruct.generalConfig.isyinshuajiance);
	ui->rbtn_zhinengcaiqie->setChecked(globalStruct.generalConfig.iszhinengcaiqie);

	if (globalStruct.generalConfig.isyinshuajiance)
	{
		globalStruct.removeState = RemoveState::PrintingInspection;
	}
	if (globalStruct.generalConfig.iszhinengcaiqie)
	{
		globalStruct.removeState = RemoveState::SmartCrop;
	}

	// 默认白色袋
	ui->btn_daizizhonglei->setText("白色袋");

	// 去掉标题栏
	this->setWindowFlags(Qt::FramelessWindowHint);

	// 开机默认关闭调试模式
	ui->ckb_debug->setChecked(false);
	globalStruct.generalConfig.isDebug = false;

	globalStruct.buildImageSaveEngine();

	// 初始化图像查看器
	_picturesViewer = new PictureViewerThumbnails(this);

	this->_carouselWidget = new CarouselWidget(this);
	auto layout=ui->gBox_main->layout();
	layout->replaceWidget(ui->pushButton, _carouselWidget);
	_carouselWidget->setSize(10);
	delete ui->pushButton;
}

void SmartCroppingOfBags::build_DlgProductSetData()
{
	_dlgProductSet = new DlgProductSetSmartCroppingOfBags(this);
}

void SmartCroppingOfBags::build_DlgProductScore()
{
	_dlgProductScore = new DlgProductScoreSmartCroppingOfBags(this);
}

void SmartCroppingOfBags::build_imageProcessorModule()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	QDir dir;

	QString enginePathFull = globalPath.modelPath;

	QFileInfo engineFile(enginePathFull);

	if (!engineFile.exists()) {
		QMessageBox::critical(this, "Error", "Engine file or Name file does not exist. The application will now exit.");
		QApplication::quit();
		return;
	}

	globalStruct.buildImageProcessorModules(enginePathFull);

	QObject::connect(globalStruct.modelCamera1.get(), &ImageProcessingModuleSmartCroppingOfBags::imageReady, this, &SmartCroppingOfBags::onCamera1Display);
	QObject::connect(globalStruct.modelCamera2.get(), &ImageProcessingModuleSmartCroppingOfBags::imageReady, this, &SmartCroppingOfBags::onCamera2Display);

}

void SmartCroppingOfBags::build_imageSaveEngine()
{
	QDir dir;
	QString imageSavePath = globalPath.imageSaveRootPath;
	//清理旧的数据

	//获取当前日期并设置保存路径
	QString currentDate = QDate::currentDate().toString("yyyy_MM_dd");
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	globalStruct.buildImageSaveEngine();
	QString imageSaveEnginePath = imageSavePath + currentDate;

	QString imagesFilePathFilePathFull = dir.absoluteFilePath(imageSaveEnginePath);
	globalStruct.imageSaveEngine->setRootPath(imagesFilePathFilePathFull);
	globalStruct.imageSaveEngine->startEngine();
}

void SmartCroppingOfBags::destroyComponents()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	//销毁相机
	globalStruct.destroyCamera();
	// 销毁图像处理模块
	//globalStruct.destroyImageProcessingModule();
	// 销毁图像保存模块
	globalStruct.destroyImageSaveEngine();
	// 销毁异步剔废线程
	//globalStruct.destroy_DetachDefectThreadZipper();
	globalStruct.destroy_MonitorIOSmartCroppingOfBags();

	//销毁板卡
	destroy_motion();
	// 销毁剔废优先队列
	globalStruct.destroy_PriorityQueue();
	// 保存参数
	globalStruct.saveGeneralConfig();
}

void SmartCroppingOfBags::read_config()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	globalStruct.buildConfigManager(rw::oso::StorageType::Xml);

	read_config_GeneralConfig();
	read_config_ScoreConfig();
	read_config_SetConfig();
}

void SmartCroppingOfBags::read_config_GeneralConfig()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

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
		globalStruct.generalConfig = cdm::GeneralConfigSmartCroppingOfBags();
		globalStruct.storeContext->save(globalStruct.generalConfig, globalPath.generalConfigPath.toStdString());
		return;
	}
	else
	{
		globalStruct.generalConfig = *globalStruct.storeContext->load(globalPath.generalConfigPath.toStdString());
	}
}

void SmartCroppingOfBags::read_config_ScoreConfig()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

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
		globalStruct.scoreConfig = cdm::ScoreConfigSmartCroppingOfBags();
		globalStruct.storeContext->save(globalStruct.scoreConfig, globalPath.scoreConfigPath.toStdString());
		return;
	}
	else
	{
		globalStruct.scoreConfig = *globalStruct.storeContext->load(globalPath.scoreConfigPath.toStdString());
	}
}

void SmartCroppingOfBags::read_config_SetConfig()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
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
		globalStruct.setConfig = cdm::SetConfigSmartCroppingOfBags();
		globalStruct.storeContext->save(globalStruct.setConfig, globalPath.setConfigPath.toStdString());
		return;
	}
	else
	{
		globalStruct.setConfig = *globalStruct.storeContext->load(globalPath.setConfigPath.toStdString());
	}
}

void SmartCroppingOfBags::btn_close_clicked()
{
	this->close();
}

void SmartCroppingOfBags::btn_pingbiquyu_clicked()
{
}

void SmartCroppingOfBags::btn_chanliangqingling_clicked()
{
}

void SmartCroppingOfBags::btn_daizizhonglei_clicked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	// 0:白色袋
	// 1:颜色袋
	if (generalConfig.daizizhonglei == 0)
	{
		generalConfig.daizizhonglei = 1;
		ui->btn_daizizhonglei->setText("颜色袋");
	}
	else if (generalConfig.daizizhonglei == 1)
	{
		generalConfig.daizizhonglei = 0;
		ui->btn_daizizhonglei->setText("白色袋");
	}
}

void SmartCroppingOfBags::btn_down_clicked()
{
}

void SmartCroppingOfBags::btn_up_clicked()
{
}

void SmartCroppingOfBags::btn_baoguang_clicked()
{
}

void SmartCroppingOfBags::btn_normalParam_clicked()
{
	_dlgProductScore->setFixedSize(this->width(), this->height());
	_dlgProductScore->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgProductScore->exec();
}

void SmartCroppingOfBags::btn_setParam_clicked()
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


void SmartCroppingOfBags::ckb_tifei_checked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.istifei = ui->rbtn_removeFunc->isChecked();

	if (generalConfig.istifei)
	{
		ui->ckb_debug->setChecked(false);
		generalConfig.isDebug = false;
	}
}

void SmartCroppingOfBags::ckb_Debug_checked(bool checked)
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.isDebug = ui->ckb_debug->isChecked();

	auto isRuning = ui->rbtn_removeFunc->isChecked();

	auto& GlobalStructData = GlobalStructDataSmartCroppingOfBags::getInstance();
	if (!isRuning) {
		if (checked) {
			GlobalStructData.setCameraDebugMod(); // 设置相机为实时采集
			GlobalStructData.runningState = RunningState::Debug;
			ui->ckb_cuntu->setChecked(false);
		}
		else {
			GlobalStructData.setCameraDefectMod(); // 重置相机为硬件触发
			GlobalStructData.runningState = RunningState::Stop;
		}
	}
	else {
		ui->ckb_debug->setChecked(false);
	}
}

void SmartCroppingOfBags::ckb_cuntu_checked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.iscuntu = ui->ckb_cuntu->isChecked();
}

void SmartCroppingOfBags::rbtn_zhinengcaiqie_clicked(bool checked)
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.isyinshuajiance = false;
	generalConfig.iszhinengcaiqie = checked;
}

void SmartCroppingOfBags::rbtn_yinshuazhiliangjiance_clicked(bool checked)
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.iszhinengcaiqie = false;
	generalConfig.isyinshuajiance = checked;
}


void SmartCroppingOfBags::updateCameraLabelState(int cameraIndex, bool state)
{
	switch (cameraIndex)
	{
	case 1:
		if (state) {
			ui->lb_xiangjilianjiezhuangtai->setText("连接成功");
			ui->lb_xiangjilianjiezhuangtai->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
		}
		else {
			ui->lb_xiangjilianjiezhuangtai->setText("连接失败");
			ui->lb_xiangjilianjiezhuangtai->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
			rw::rqw::WarningInfo info;
			info.message = "相机1断连";
			info.type = rw::rqw::WarningType::Error;
			info.warningId = WarningId::ccameraDisconnectAlarm1;
			//labelWarning->addWarning(info);
		}
		break;
	//case 2:
	//	if (state) {
	//		ui->label_camera2State->setText("连接成功");
	//		ui->label_camera2State->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
	//	}
	//	else {
	//		ui->label_camera2State->setText("连接失败");
	//		ui->label_camera2State->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
	//		rw::rqw::WarningInfo info;
	//		info.message = "相机2断连";
	//		info.type = rw::rqw::WarningType::Error;
	//		info.warningId = WarningId::ccameraDisconnectAlarm2;
	//		//labelWarning->addWarning(info);
	//	}
	//	break;
	default:
		break;
	}
}

void SmartCroppingOfBags::onCamera1Display(QPixmap image)
{
	ui->lb_show->setPixmap(image.scaled(ui->lb_show->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void SmartCroppingOfBags::onCamera2Display(QPixmap image)
{

}

void SmartCroppingOfBags::onCameraNGDisplay(QPixmap image, size_t index, bool isbad)
{
	if (isbad)
	{
		if (index == 1)
		{
			ui->lb_Ngshow->setPixmap(image.scaled(ui->lb_Ngshow->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		}
		else if (index == 2)
		{
			//ui->label_imgDisplay_4->setPixmap(image.scaled(ui->label_imgDisplay_4->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		}
	}
}

void SmartCroppingOfBags::updateCardLabelState(bool state)
{
	if (state) {
		ui->lb_bankalianjiezhuangtai->setText("连接成功");
		ui->lb_bankalianjiezhuangtai->setStyleSheet(QString("QLabel{color:rgb(0, 230, 0);} "));
	}
	else {
		ui->lb_bankalianjiezhuangtai->setText("连接失败");
		ui->lb_bankalianjiezhuangtai->setStyleSheet(QString("QLabel{color:rgb(230, 0, 0);} "));
	}
}
