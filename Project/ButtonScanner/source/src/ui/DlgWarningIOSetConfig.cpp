#include "DlgWarningIOSetConfig.h"

#include"GlobalStruct.h"

DlgWarningIOSetConfig::DlgWarningIOSetConfig(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningIOSetConfigClass())
{
	ui->setupUi(this);

	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	build_ui();
	read_config();
	build_connect();
}

DlgWarningIOSetConfig::~DlgWarningIOSetConfig()
{
	delete ui;
}

void DlgWarningIOSetConfig::build_ui()
{

}

void DlgWarningIOSetConfig::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_exit_clicked);
}

void DlgWarningIOSetConfig::read_config()
{
	auto& config = GlobalStructData::getInstance().warningIOSetConfig;
	ui->pbtn_DIAirPressureValue->setText(QString::number(config.DIAirPressure));
	ui->pbtn_DICameraTriggerValue1->setText(QString::number(config.DICameraTrigger1));
	ui->pbtn_DICameraTriggerValue2->setText(QString::number(config.DICameraTrigger2));
	ui->pbtn_DICameraTriggerValue3->setText(QString::number(config.DICameraTrigger3));
	ui->pbtn_DICameraTriggerValue4->setText(QString::number(config.DICameraTrigger4));
	ui->pbtn_DIShutdownComputerValue->setText(QString::number(config.DIShutdownComputer));
	ui->pbtn_DIStartValue->setText(QString::number(config.DIStart));
	ui->pbtn_DIStopValue->setText(QString::number(config.DIStop));
	ui->pbtn_DOBlowTime1Value->setText(QString::number(config.DOBlow1));
	ui->pbtn_DOBlowTime2Value->setText(QString::number(config.DOBlow2));
	ui->pbtn_DOBlowTime3Value->setText(QString::number(config.DOBlow3));
	ui->pbtn_DOBlowTime4Value->setText(QString::number(config.DOBlow4));
	ui->pbtn_DOMotoPowerValue->setText(QString::number(config.DOMotoPower));
	ui->pbtn_DOGreenLightValue->setText(QString::number(config.DOGreenLight));
	ui->pbtn_DORedLightValue->setText(QString::number(config.DORedLight));
	ui->pbtn_DOSideLightValue->setText(QString::number(config.DOSideLight));
	ui->pbtn_DOUpLightValue->setText(QString::number(config.DOUpLight));
	ui->pbtn_DODownLightValue->setText(QString::number(config.DODownLight));
	ui->pbtn_DOStoreLightValue->setText(QString::number(config.DOStrobeLight));
}

void DlgWarningIOSetConfig::pbtn_exit_clicked()
{
	this->close();
}
