#include "ZipperScanner.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "Utilty.hpp"

ZipperScanner::ZipperScanner(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::ZipperScannerClass())
{
	ui->setupUi(this);

	read_config();

	build_ui();

	build_connect();
}

ZipperScanner::~ZipperScanner()
{
	delete ui;
}

void ZipperScanner::build_ui()
{
	build_ZipperScannerData();
	build_DlgProductSetData();


}

void ZipperScanner::build_connect()
{
}

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

void ZipperScanner::build_DlgProductSetData()
{
	_dlgProductSet = new DlgProductSet(this);
}

void ZipperScanner::read_config()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	globalStruct.buildConfigManager(rw::oso::StorageType::Xml);

	read_config_GeneralConfig();
	read_config_ScoreConfig();
	read_config_SetConfig();

}

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

