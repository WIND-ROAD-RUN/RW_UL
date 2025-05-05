#include "stdafx.h"
#include "DlgExposureTimeSet.h"

#include"GlobalStruct.h"
#include"NumberKeyboard.h"

DlgExposureTimeSet::DlgExposureTimeSet(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgExposureTimeSetClass())
{
	ui->setupUi(this);
	build_connect();
	build_ui();
}

DlgExposureTimeSet::~DlgExposureTimeSet()
{
	ResetCamera();
	delete ui;
}

void DlgExposureTimeSet::build_ui()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& exposureTimeSetConfig = globalStruct.dlgExposureTimeSetConfig;
	ui->pbtn_exposureTimeValue->setText(QString::number(exposureTimeSetConfig.expousureTime));
}

void DlgExposureTimeSet::build_connect()
{
	QObject::connect(ui->pbtn_exposureTimeValue, &QPushButton::clicked,
		this, &DlgExposureTimeSet::pbtn_exposureTimeValue_clicked);

	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgExposureTimeSet::pbtn_close_clicked);
}

void DlgExposureTimeSet::SetCamera()
{
	auto& globalStruct = GlobalStructData::getInstance();

	if (globalStruct.camera1) {
		globalStruct.camera1->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera1->setFrameRate(5);
	}

	if (globalStruct.camera2) {
		globalStruct.camera2->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera2->setFrameRate(5);
	}

	if (globalStruct.camera3) {
		globalStruct.camera3->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera3->setFrameRate(5);
	}

	if (globalStruct.camera4) {
		globalStruct.camera4->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera4->setFrameRate(5);
	}
}

void DlgExposureTimeSet::ResetCamera()
{
	auto& globalStruct = GlobalStructData::getInstance();

	if (globalStruct.camera1)
	{
		globalStruct.camera1->setTriggerMode(rw::rqw::CameraObjectTrigger::Hardware);
		globalStruct.camera1->setFrameRate(40);
	}
	if (globalStruct.camera2)
	{
		globalStruct.camera2->setTriggerMode(rw::rqw::CameraObjectTrigger::Hardware);
		globalStruct.camera2->setFrameRate(40);
	}
	if (globalStruct.camera3)
	{
		globalStruct.camera3->setTriggerMode(rw::rqw::CameraObjectTrigger::Hardware);
		globalStruct.camera3->setFrameRate(40);
	}
	if (globalStruct.camera4)
	{
		globalStruct.camera4->setTriggerMode(rw::rqw::CameraObjectTrigger::Hardware);
		globalStruct.camera4->setFrameRate(40);
	}
}

void DlgExposureTimeSet::pbtn_exposureTimeValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto newValue = numKeyBord.getValue().toInt();
		if (newValue < 10) {
			QMessageBox::warning(this, "错误", "曝光时间范围应设置为10~700");
			return;
		}
		if (newValue > 700) {
			QMessageBox::warning(this, "错误", "曝光时间范围应设置为100~700");
			return;
		}

		auto& globalStruct = GlobalStructData::getInstance();
		ui->pbtn_exposureTimeValue->setText(QString::number(newValue));
		globalStruct.dlgExposureTimeSetConfig.expousureTime = newValue;

		globalStruct.setCameraExposureTime(1, newValue);
		globalStruct.setCameraExposureTime(2, newValue);
		globalStruct.setCameraExposureTime(3, newValue);
		globalStruct.setCameraExposureTime(4, newValue);

		globalStruct.saveDlgExposureTimeSetConfig();
	}
}

void DlgExposureTimeSet::pbtn_close_clicked()
{
	this->close();
}