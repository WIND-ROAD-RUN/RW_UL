#include "SmartCroppingOfBags.h"

#include "oso_StorageContext.hpp"
#include "ui_SmartCroppingOfBags.h"
#include <GlobalStruct.hpp>
#include <QMessageBox>

#include "NumberKeyboard.h"

SmartCroppingOfBags::SmartCroppingOfBags(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::SmartCroppingOfBagsClass())
{
	ui->setupUi(this);

	// 读取参数
	read_config();

	// 构建UI
	build_ui();

	// 构建优先队列
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	globalStruct.build_PriorityQueue();

	// 构建异步剔废线程
	globalStruct.build_DetachDefectThreadSmartCroppingOfBags();

	// 构建图像保存引擎
	build_imageSaveEngine();

	// 构建图像处理模块
	build_imageProcessorModule();

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
	QObject::connect(ui->ckb_zhinengcaiqie, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_zhinengcaiqie_checked);
	QObject::connect(ui->ckb_tifei, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_tifei_checked);
	QObject::connect(ui->ckb_huikan, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_huikan_checked);
	QObject::connect(ui->ckb_yinshuazhiliangjiance, &QCheckBox::clicked,
		this, &SmartCroppingOfBags::ckb_yinshuazhiliangjiance_checked);
	QObject::connect(ui->btn_close, &QPushButton::clicked,
		this, &SmartCroppingOfBags::btn_close_clicked);
}

void SmartCroppingOfBags::build_SmartCroppingOfBagsData()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& smartCroppingOfBagsConfig = globalStruct.generalConfig;

	// 初始化全局数据
	// 加载主窗体UI的设置
	ui->ckb_zhinengcaiqie->setChecked(globalStruct.generalConfig.iszhinengcaiqie);
	ui->lb_shengchanzongliang->setText(QString::number(globalStruct.generalConfig.shengchanzongliang));
	ui->lb_lianglv->setText(QString::number(globalStruct.generalConfig.shengchanlianglv));
	ui->lb_feipinshuliang->setText(QString::number(globalStruct.generalConfig.feipinshuliang));
	ui->lb_pingjundaichang->setText(QString::number(globalStruct.generalConfig.pingjundaichang));
	ui->ckb_tifei->setChecked(globalStruct.generalConfig.istifei);
	ui->ckb_huikan->setChecked(globalStruct.generalConfig.ishuikan);
	ui->ckb_yinshuazhiliangjiance->setChecked(globalStruct.generalConfig.isyinshuazhiliangjiance);
	ui->btn_baoguang->setText(QString::number(globalStruct.generalConfig.baoguang));
	// 默认白色袋
	ui->btn_daizizhonglei->setText("白色袋");

	// 去掉标题栏
	this->setWindowFlags(Qt::FramelessWindowHint);

	// 暂不启用"印刷质量检测"
	ui->ckb_yinshuazhiliangjiance->setVisible(false);
	ui->ckb_yinshuazhiliangjiance->setChecked(false);
	smartCroppingOfBagsConfig.isyinshuazhiliangjiance = false;

	globalStruct.buildImageSaveEngine();

	// 初始化图像查看器
	_picturesViewer = new PictureViewerThumbnails(this);
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
	// 销毁剔废优先队列
	//globalStruct.destroy_PriorityQueue();
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

void SmartCroppingOfBags::ckb_zhinengcaiqie_checked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.iszhinengcaiqie = ui->ckb_zhinengcaiqie->isChecked();
}

void SmartCroppingOfBags::ckb_tifei_checked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.istifei = ui->ckb_tifei->isChecked();
}

void SmartCroppingOfBags::ckb_huikan_checked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.ishuikan = ui->ckb_huikan->isChecked();
}

void SmartCroppingOfBags::ckb_yinshuazhiliangjiance_checked()
{
	auto& generalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().generalConfig;
	generalConfig.isyinshuazhiliangjiance = ui->ckb_yinshuazhiliangjiance->isChecked();
}
