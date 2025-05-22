#include "stdafx.h"
#include "DlgProduceLineSet.h"

#include"NumberKeyboard.h"
#include"GlobalStruct.h"
#include"scc_motion.h"
#include "rqw_CameraObject.hpp"

DlgProduceLineSet::DlgProduceLineSet(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgProduceLineSetClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

DlgProduceLineSet::~DlgProduceLineSet()
{
	monitorIoStateThread->destroyThread();
	delete ui;
}

void DlgProduceLineSet::showEvent(QShowEvent* show_event)
{
	QDialog::showEvent(show_event);
	monitorIoStateThread->setRunning(true);
}

void DlgProduceLineSet::build_ui()
{
	read_config();
	monitorIoStateThread = new MonitorIOStateThread(this);
	monitorIoStateThread->setRunning(false);
	monitorIoStateThread->start();
	dlgWarningManager = new DlgWarningManager(this);
	connect(monitorIoStateThread, &MonitorIOStateThread::DIState,
		this, &DlgProduceLineSet::onDIState);
	connect(monitorIoStateThread, &MonitorIOStateThread::DOState,
		this, &DlgProduceLineSet::onDOState);
}

void DlgProduceLineSet::read_config()
{
	auto& globalStruct = GlobalStructData::getInstance();
	//SetValue
	ui->pbtn_blowDistance1->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowDistance1));
	ui->pbtn_blowDistance2->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowDistance2));
	ui->pbtn_blowDistance3->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowDistance3));
	ui->pbtn_blowDistance4->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowDistance4));
	ui->pbtn_blowTime1->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowTime1));
	ui->pbtn_blowTime2->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowTime2));
	ui->pbtn_blowTime3->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowTime3));
	ui->pbtn_blowTime4->setText(QString::number(globalStruct.dlgProduceLineSetConfig.blowTime4));
	ui->pbtn_pixelEquivalent1->setText(QString::number(globalStruct.dlgProduceLineSetConfig.pixelEquivalent1));
	ui->pbtn_pixelEquivalent2->setText(QString::number(globalStruct.dlgProduceLineSetConfig.pixelEquivalent2));
	ui->pbtn_pixelEquivalent3->setText(QString::number(globalStruct.dlgProduceLineSetConfig.pixelEquivalent3));
	ui->pbtn_pixelEquivalent4->setText(QString::number(globalStruct.dlgProduceLineSetConfig.pixelEquivalent4));
	ui->pbtn_limit1->setText(QString::number(globalStruct.dlgProduceLineSetConfig.limit1));
	ui->pbtn_limit2->setText(QString::number(globalStruct.dlgProduceLineSetConfig.limit2));
	ui->pbtn_limit3->setText(QString::number(globalStruct.dlgProduceLineSetConfig.limit3));
	ui->pbtn_limit4->setText(QString::number(globalStruct.dlgProduceLineSetConfig.limit4));
	ui->cbox_DO7->setChecked(globalStruct.dlgProduceLineSetConfig.powerOn);
	ui->cbox_DO1->setChecked(globalStruct.dlgProduceLineSetConfig.blowingEnable1);
	ui->cbox_DO2->setChecked(globalStruct.dlgProduceLineSetConfig.blowingEnable2);
	ui->cbox_DO3->setChecked(globalStruct.dlgProduceLineSetConfig.blowingEnable3);
	ui->cbox_DO4->setChecked(globalStruct.dlgProduceLineSetConfig.blowingEnable4);
	ui->cbox_DO6->setChecked(globalStruct.dlgProduceLineSetConfig.none);
	ui->cbox_DO8->setChecked(globalStruct.dlgProduceLineSetConfig.run);
	ui->cbox_DO9->setChecked(globalStruct.dlgProduceLineSetConfig.alarm);
	ui->cbox_workstationProtection12->setChecked(globalStruct.dlgProduceLineSetConfig.workstationProtection12);
	ui->cbox_workstationProtection34->setChecked(globalStruct.dlgProduceLineSetConfig.workstationProtection34);
	ui->pbtn_motorSpeed->setText(QString::number(globalStruct.dlgProduceLineSetConfig.motorSpeed));
	ui->pbtn_beltReductionRatio->setText(QString::number(globalStruct.dlgProduceLineSetConfig.beltReductionRatio));
	ui->pbtn_accelerationAndDeceleration->setText(QString::number(globalStruct.dlgProduceLineSetConfig.accelerationAndDeceleration));
	ui->pbtn_minBrightness->setText(QString::number(globalStruct.dlgProduceLineSetConfig.minBrightness));
	ui->pbtn_maxBrightness->setText(QString::number(globalStruct.dlgProduceLineSetConfig.maxBrightness));
	ui->pbtn_codeWheel->setText(QString::number(globalStruct.dlgProduceLineSetConfig.codeWheel));
	ui->pbtn_pulseFactor->setText(QString::number(globalStruct.dlgProduceLineSetConfig.pulseFactor));
	
	//Deprecated widget
	ui->pbtn_maxBrightness->setVisible(false);
	ui->pbtn_minBrightness->setVisible(false);
	ui->label_lightRange->setVisible(false);
	ui->label_lightRange1->setVisible(false);
	//Deprecated widget
}

void DlgProduceLineSet::build_connect()
{
	QObject::connect(ui->pbtn_blowDistance1, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowDistance1_clicked);
	QObject::connect(ui->pbtn_blowDistance2, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowDistance2_clicked);
	QObject::connect(ui->pbtn_blowDistance3, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowDistance3_clicked);
	QObject::connect(ui->pbtn_blowDistance4, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowDistance4_clicked);
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_close_clicked);

	QObject::connect(ui->pbtn_blowTime1, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowTime1_clicked);
	QObject::connect(ui->pbtn_blowTime2, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowTime2_clicked);
	QObject::connect(ui->pbtn_blowTime3, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowTime3_clicked);
	QObject::connect(ui->pbtn_blowTime4, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_blowTime4_clicked);

	QObject::connect(ui->pbtn_pixelEquivalent1, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_pixelEquivalent1_clicked);
	QObject::connect(ui->pbtn_pixelEquivalent2, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_pixelEquivalent2_clicked);
	QObject::connect(ui->pbtn_pixelEquivalent3, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_pixelEquivalent3_clicked);
	QObject::connect(ui->pbtn_pixelEquivalent4, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_pixelEquivalent4_clicked);

	QObject::connect(ui->pbtn_limit1, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_limit1_clicked);
	QObject::connect(ui->pbtn_limit2, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_limit2_clicked);
	QObject::connect(ui->pbtn_limit3, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_limit3_clicked);
	QObject::connect(ui->pbtn_limit4, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_limit4_clicked);

	QObject::connect(ui->pbtn_minBrightness, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_minBrightness_clicked);
	QObject::connect(ui->pbtn_maxBrightness, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_maxBrightness_clicked);

	QObject::connect(ui->pbtn_motorSpeed, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_motorSpeed_clicked);
	QObject::connect(ui->pbtn_beltReductionRatio, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_beltReductionRatio_clicked);
	QObject::connect(ui->pbtn_accelerationAndDeceleration, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_accelerationAndDeceleration_clicked);

	QObject::connect(ui->cbox_workstationProtection12, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_workstationProtection12_checked);
	QObject::connect(ui->cbox_workstationProtection34, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_workstationProtection34_checked);
	QObject::connect(ui->cbox_debugMode, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_debugMode_checked);

	QObject::connect(ui->pbtn_codeWheel, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_codeWheel_clicked);
	QObject::connect(ui->pbtn_pulseFactor, &QPushButton::clicked,
		this, &DlgProduceLineSet::pbtn_pulseFactor_clicked);

	QObject::connect(ui->cBox_takeMaskPictures, &QCheckBox::checkStateChanged,
		this, &DlgProduceLineSet::cBox_takeMaskPictures);
	QObject::connect(ui->cBox_takeNgPictures, &QCheckBox::checkStateChanged,
		this, &DlgProduceLineSet::cBox_takeNgPictures);
	QObject::connect(ui->cBox_takeOkPictures, &QCheckBox::checkStateChanged,
		this, &DlgProduceLineSet::cBox_takeOkPictures);

	QObject::connect(ui->rbtn_drawCircle, &QRadioButton::clicked,
		this, &DlgProduceLineSet::rbtn_drawCircle_clicked);
	QObject::connect(ui->rbtn_drawRec, &QRadioButton::clicked,
		this, &DlgProduceLineSet::rbtn_drawRectangle_clicked);

	QObject::connect(ui->cbox_DO0, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO0_checked);
	QObject::connect(ui->cbox_DO1, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO1_checked);
	QObject::connect(ui->cbox_DO2, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO2_checked);
	QObject::connect(ui->cbox_DO3, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO3_checked);
	QObject::connect(ui->cbox_DO4, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO4_checked);
	QObject::connect(ui->cbox_DO5, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO5_checked);
	QObject::connect(ui->cbox_DO6, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO6_checked);
	QObject::connect(ui->cbox_DO7, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO7_checked);
	QObject::connect(ui->cbox_DO8, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO8_checked);
	QObject::connect(ui->cbox_DO9, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO9_checked);
	QObject::connect(ui->cbox_DO10, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DO10_checked);
	QObject::connect(ui->cbox_beltControl, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_beltControl);

	QObject::connect(ui->cbox_DI0, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI0_checked);
	QObject::connect(ui->cbox_DI1, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI1_checked);
	QObject::connect(ui->cbox_DI2, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI2_checked);
	QObject::connect(ui->cbox_DI3, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI3_checked);
	QObject::connect(ui->cbox_DI4, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI4_checked);
	QObject::connect(ui->cbox_DI5, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI5_checked);
	QObject::connect(ui->cbox_DI6, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI6_checked);
	QObject::connect(ui->cbox_DI7, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI7_checked);
	QObject::connect(ui->cbox_DI8, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI8_checked);
	QObject::connect(ui->cbox_DI9, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI9_checked);
	QObject::connect(ui->cbox_DI10, &QCheckBox::clicked,
		this, &DlgProduceLineSet::cbox_DI10_checked);

	QObject::connect(ui->pbtn_warnManager, &QPushButton::clicked
		, this, &DlgProduceLineSet::pbtn_warningManager_clicked);
}

float DlgProduceLineSet::get_blowTime()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	auto outsideDiameterValue = GlobalStructData.dlgProductSetConfig.outsideDiameterValue;
	auto beltSpeed = GlobalStructData.dlgProduceLineSetConfig.motorSpeed;
	auto blowTime = outsideDiameterValue / beltSpeed * 1000 / 2;
	GlobalStructData.dlgProductSetConfig.blowTime = blowTime;
	return blowTime;
}

void DlgProduceLineSet::updateBeltSpeed()
{
	ui->pbtn_motorSpeed->setText(QString::number(GlobalStructData::getInstance().dlgProduceLineSetConfig.motorSpeed));
}

void DlgProduceLineSet::pbtn_blowDistance1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowDistance1->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowDistance1 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowTime1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowTime1->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowTime1 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowDistance2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowDistance2->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowDistance2 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowTime2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowTime2->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowTime2 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowDistance3_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowDistance3->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowDistance3 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowTime3_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowTime3->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowTime3 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowDistance4_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowDistance4->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowDistance4 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_blowTime4_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowTime4->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.blowTime4 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_pixelEquivalent1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_pixelEquivalent1->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.pixelEquivalent1 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_pixelEquivalent2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_pixelEquivalent2->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.pixelEquivalent2 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_pixelEquivalent3_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_pixelEquivalent3->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.pixelEquivalent3 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_pixelEquivalent4_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_pixelEquivalent4->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.pixelEquivalent4 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_limit1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_limit1->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.limit1 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_limit2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_limit2->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.limit2 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_limit3_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_limit3->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.limit3 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_limit4_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_limit4->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.limit4 = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_minBrightness_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_minBrightness->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.minBrightness = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_maxBrightness_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_maxBrightness->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.maxBrightness = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_motorSpeed_clicked()
{
	auto& globalStrut = GlobalStructData::getInstance();
	auto currentRunningState = globalStrut.runningState.load();
	if (currentRunningState == RunningState::OpenRemoveFunc)
	{
		QMessageBox::warning(this, "错误", "请先停止生产线");
		return;
	}
	else
	{
		NumberKeyboard numKeyBord;
		numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
		auto isAccept = numKeyBord.exec();
		if (isAccept == QDialog::Accepted)
		{
			auto value = numKeyBord.getValue();
			auto& GlobalStructData = GlobalStructData::getInstance();
			ui->pbtn_motorSpeed->setText(value);
			GlobalStructData.dlgProduceLineSetConfig.motorSpeed = value.toDouble();
		}

		get_blowTime();
	}
}

void DlgProduceLineSet::pbtn_beltReductionRatio_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_beltReductionRatio->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.beltReductionRatio = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_accelerationAndDeceleration_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_accelerationAndDeceleration->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.accelerationAndDeceleration = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_codeWheel_clicked() {
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_codeWheel->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.codeWheel = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_pulseFactor_clicked() {
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_pulseFactor->setText(value);
		GlobalStructData.dlgProduceLineSetConfig.pulseFactor = value.toDouble();
	}
}

void DlgProduceLineSet::pbtn_close_clicked()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.saveDlgProduceLineSetConfig();
	this->close();
}

void DlgProduceLineSet::cbox_DO7_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::warnUpLightOut, ischeck);
}

void DlgProduceLineSet::cbox_DO1_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut( ControlLines::blowLine1.ioNum, ischeck);
}

void DlgProduceLineSet::cbox_DO2_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut( ControlLines::blowLine2.ioNum, ischeck);
}

void DlgProduceLineSet::cbox_DO3_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut( ControlLines::blowLine3.ioNum, ischeck);
}

void DlgProduceLineSet::cbox_DO4_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::blowLine4.ioNum, ischeck);
}

void DlgProduceLineSet::cbox_DO5_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::warnGreenOut, ischeck);
}

void DlgProduceLineSet::cbox_DO6_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::warnRedOut, ischeck);
}

void DlgProduceLineSet::cbox_DO8_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::warnSideLightOut, ischeck);
}

void DlgProduceLineSet::cbox_DO9_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::warnDownLightOut, ischeck);
}

void DlgProduceLineSet::cbox_DO10_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
}

void DlgProduceLineSet::cbox_beltControl(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	if (ischeck)
	{
		zwy::scc::GlobalMotion::getInstance().motionPtr.get()->AxisRun(ControlLines::beltAsis, -1);
	}
	else
	{
		zwy::scc::GlobalMotion::getInstance().motionPtr.get()->StopAllAxis();

	}

	
}

void DlgProduceLineSet::cbox_DI0_checked(bool ischeck)
{
	ui->cbox_DI0->setChecked(false);
}

void DlgProduceLineSet::cbox_DI1_checked(bool ischeck)
{
	ui->cbox_DI1->setChecked(false);
}

void DlgProduceLineSet::cbox_DI2_checked(bool ischeck)
{
	ui->cbox_DI2->setChecked(false);
}

void DlgProduceLineSet::cbox_DI3_checked(bool ischeck)
{
	ui->cbox_DI3->setChecked(false);
}

void DlgProduceLineSet::cbox_DI4_checked(bool ischeck)
{
	ui->cbox_DI4->setChecked(false);
}

void DlgProduceLineSet::cbox_DI5_checked(bool ischeck)
{
	ui->cbox_DI5->setChecked(false);
}

void DlgProduceLineSet::cbox_DI6_checked(bool ischeck)
{
	ui->cbox_DI6->setChecked(false);
}

void DlgProduceLineSet::cbox_DI7_checked(bool ischeck)
{
	ui->cbox_DI7->setChecked(false);
}

void DlgProduceLineSet::cbox_DI8_checked(bool ischeck)
{
	ui->cbox_DI8->setChecked(false);
}

void DlgProduceLineSet::cbox_DI9_checked(bool ischeck)
{
	ui->cbox_DI9->setChecked(false);
}

void DlgProduceLineSet::cbox_DI10_checked(bool ischeck)
{
	ui->cbox_DI10->setChecked(false);
}

void DlgProduceLineSet::onDIState(int index, bool state)
{
	switch (index)
	{
	case ControlLines::stopIn:
		ui->cbox_DI2->setChecked(state);
		break;
	case ControlLines::startIn:
		ui->cbox_DI1->setChecked(state);
		break;
	case ControlLines::airWarnIN:
		ui->cbox_DI4->setChecked(state);
		break;
	case ControlLines::shutdownComputerIn:
		ui->cbox_DI3->setChecked(state);
		break;
	case ControlLines::camer1In:
		ui->cbox_DI5->setChecked(state);
		break;
	case ControlLines::camer2In:
		ui->cbox_DI6->setChecked(state);
		break;
	case ControlLines::camer3In:
		ui->cbox_DI7->setChecked(state);
		break;
	case ControlLines::camer4In:
		ui->cbox_DI8->setChecked(state);
		break;
	default:
		break;
	}
}

void DlgProduceLineSet::onDOState(int index, bool state)
{
	if (index == ControlLines::motoPowerOut) {
		ui->cbox_DO0->setChecked(state);
	}
	else if (index == ControlLines::blowLine1.ioNum) {
		ui->cbox_DO1->setChecked(state);
	}
	else if (index == ControlLines::blowLine2.ioNum) {
		ui->cbox_DO2->setChecked(state);
	}
	else if (index == ControlLines::blowLine3.ioNum) {
		ui->cbox_DO3->setChecked(state);
	}
	else if (index == ControlLines::blowLine4.ioNum) {
		ui->cbox_DO4->setChecked(state);
	}
	else if (index == ControlLines::warnGreenOut) {
		ui->cbox_DO5->setChecked(state);
	}
	else if (index == ControlLines::warnRedOut) {
		ui->cbox_DO6->setChecked(state);
	}
	else if (index == ControlLines::warnUpLightOut) {
		ui->cbox_DO7->setChecked(state);
	}
	else if (index == ControlLines::warnSideLightOut) {
		ui->cbox_DO8->setChecked(state);
	}
	else if (index == ControlLines::warnDownLightOut) {
		ui->cbox_DO9->setChecked(state);
	}
}

void DlgProduceLineSet::pbtn_warningManager_clicked()
{
	dlgWarningManager->show();
}

void DlgProduceLineSet::cbox_workstationProtection12_checked(bool ischeck)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.workstationProtection12 = ischeck;
}

void DlgProduceLineSet::cbox_workstationProtection34_checked(bool ischeck)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.workstationProtection34 = ischeck;
}

void DlgProduceLineSet::cbox_debugMode_checked(bool ischeck)
{
	isDebug = ischeck;
	monitorIoStateThread->setRunning(!ischeck);
}

void DlgProduceLineSet::cBox_takeMaskPictures(bool ischeck)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.takeMaskPictures = ischeck;
}

void DlgProduceLineSet::cBox_takeNgPictures(bool ischeck)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.takeNgPictures = ischeck;
}

void DlgProduceLineSet::cBox_takeOkPictures(bool ischeck)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.takeOkPictures = ischeck;
}

void DlgProduceLineSet::rbtn_drawCircle_clicked()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.drawCircle = true;
	GlobalStructData.dlgProduceLineSetConfig.drawRec = false;
}

void DlgProduceLineSet::rbtn_drawRectangle_clicked()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProduceLineSetConfig.drawCircle = false;
	GlobalStructData.dlgProduceLineSetConfig.drawRec = true;
}

void DlgProduceLineSet::cbox_DO0_checked(bool ischeck)
{
	if (!isDebug)
	{
		return;
	}
	zwy::scc::GlobalMotion::getInstance().motionPtr.get()->SetIOOut(ControlLines::motoPowerOut, ischeck);
}
