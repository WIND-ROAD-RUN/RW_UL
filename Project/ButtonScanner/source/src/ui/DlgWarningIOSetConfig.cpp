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

	//DI
	connect(ui->pbtn_DIStartValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIStartValue_clicked);
	connect(ui->pbtn_DIStopValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIStopValue_clicked);
	connect(ui->pbtn_DIShutdownComputerValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIShutdownComputerValue_clicked);
	connect(ui->pbtn_DIAirPressureValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIAirPressureValue_clicked);
	connect(ui->pbtn_DICameraTriggerValue1, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue1_clicked);
	connect(ui->pbtn_DICameraTriggerValue2, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue2_clicked);
	connect(ui->pbtn_DICameraTriggerValue3, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue3_clicked);
	connect(ui->pbtn_DICameraTriggerValue4, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue4_clicked);

	//DO
	connect(ui->pbtn_DOMotoPowerValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOMotoPowerValue_clicked);
	connect(ui->pbtn_DOBlowTime1Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime1Value_clicked);
	connect(ui->pbtn_DOBlowTime2Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime2Value_clicked);
	connect(ui->pbtn_DOBlowTime3Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime3Value_clicked);
	connect(ui->pbtn_DOBlowTime4Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime4Value_clicked);
	connect(ui->pbtn_DOGreenLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOGreenLightValue_clicked);
	connect(ui->pbtn_DORedLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DORedLightValue_clicked);
	connect(ui->pbtn_DOUpLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOUpLightValue_clicked);
	connect(ui->pbtn_DOSideLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOSideLightValue_clicked);
	connect(ui->pbtn_DODownLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DODownLightValue_clicked);
	connect(ui->pbtn_DOStoreLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOStoreLightValue_clicked);

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

void DlgWarningIOSetConfig::pbtn_DIStartValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DIStopValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DIShutdownComputerValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DIAirPressureValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue1_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue2_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue3_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue4_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime1Value_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime2Value_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime3Value_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime4Value_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOGreenLightValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DORedLightValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOUpLightValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOSideLightValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DODownLightValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOStoreLightValue_clicked()
{
}

void DlgWarningIOSetConfig::pbtn_DOMotoPowerValue_clicked()
{
}
