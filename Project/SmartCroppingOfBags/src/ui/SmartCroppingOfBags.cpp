#include "SmartCroppingOfBags.h"

#include "oso_StorageContext.hpp"
#include "ui_SmartCroppingOfBags.h"
#include <GlobalStruct.hpp>
#include <QMessageBox>

SmartCroppingOfBags::SmartCroppingOfBags(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::SmartCroppingOfBagsClass())
{
	ui->setupUi(this);
}

SmartCroppingOfBags::~SmartCroppingOfBags()
{
	delete ui;
}

void SmartCroppingOfBags::build_ui()
{

}

void SmartCroppingOfBags::build_SmartCroppingOfBagsData()
{

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
	}
}
