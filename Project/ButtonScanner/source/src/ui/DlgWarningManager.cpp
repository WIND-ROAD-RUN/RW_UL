#include "DlgWarningManager.h"

#include"GlobalStruct.h"
#include "ui_DlgWarningManager.h"

DlgWarningManager::DlgWarningManager(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningManagerClass())
{
	ui->setupUi(this);
	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	build_ui();
	build_connect();
}

DlgWarningManager::~DlgWarningManager()
{
	delete ui;
}

void DlgWarningManager::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked, this, &DlgWarningManager::pbtn_exit_clicked);
	connect(ui->cbox_cameraDisconnect1, &QCheckBox::clicked, this, &DlgWarningManager::cbox_cameraDisconnect1_clicked);
	connect(ui->cbox_cameraDisconnect2, &QCheckBox::clicked, this, &DlgWarningManager::cbox_cameraDisconnect2_clicked);
	connect(ui->cbox_cameraDisconnect3, &QCheckBox::clicked, this, &DlgWarningManager::cbox_cameraDisconnect3_clicked);
	connect(ui->cbox_cameraDisconnect4, &QCheckBox::clicked, this, &DlgWarningManager::cbox_cameraDisconnect4_clicked);
	connect(ui->cbox_workTrigger1, &QCheckBox::clicked, this, &DlgWarningManager::cbox_workTrigger1_clicked);
	connect(ui->cbox_workTrigger2, &QCheckBox::clicked, this, &DlgWarningManager::cbox_workTrigger2_clicked);
	connect(ui->cbox_workTrigger3, &QCheckBox::clicked, this, &DlgWarningManager::cbox_workTrigger3_clicked);
	connect(ui->cbox_workTrigger4, &QCheckBox::clicked, this, &DlgWarningManager::cbox_workTrigger4_clicked);
	connect(ui->cbox_airPressure, &QCheckBox::clicked, this, &DlgWarningManager::cbox_airPressure_clicked);

}

void DlgWarningManager::build_ui()
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	ui->cbox_airPressure->setChecked(config.airPressure);
	ui->cbox_cameraDisconnect1->setChecked(config.cameraDisconnect1);
	ui->cbox_cameraDisconnect2->setChecked(config.cameraDisconnect2);
	ui->cbox_cameraDisconnect3->setChecked(config.cameraDisconnect3);
	ui->cbox_cameraDisconnect4->setChecked(config.cameraDisconnect4);
	ui->cbox_workTrigger1->setChecked(config.workTrigger1);
	ui->cbox_workTrigger2->setChecked(config.workTrigger2);
	ui->cbox_workTrigger3->setChecked(config.workTrigger3);
	ui->cbox_workTrigger4->setChecked(config.workTrigger4);
}

void DlgWarningManager::pbtn_exit_clicked()
{
	this->close();
}

void DlgWarningManager::cbox_cameraDisconnect1_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.cameraDisconnect1 = checked;
}

void DlgWarningManager::cbox_cameraDisconnect2_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.cameraDisconnect2 = checked;
}

void DlgWarningManager::cbox_cameraDisconnect3_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.cameraDisconnect3 = checked;
}

void DlgWarningManager::cbox_cameraDisconnect4_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.cameraDisconnect4 = checked;
}

void DlgWarningManager::cbox_workTrigger1_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.workTrigger1 = checked;
}

void DlgWarningManager::cbox_workTrigger2_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.workTrigger2 = checked;
}

void DlgWarningManager::cbox_workTrigger3_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.workTrigger3 = checked;
}

void DlgWarningManager::cbox_workTrigger4_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.workTrigger4 = checked;
}

void DlgWarningManager::cbox_airPressure_clicked(bool checked)
{
	auto& config = GlobalStructData::getInstance().dlgWarningManagerConfig;
	config.airPressure = checked;
}

