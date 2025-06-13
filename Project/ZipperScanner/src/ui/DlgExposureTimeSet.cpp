#include "DlgExposureTimeSet.h"

#include"GlobalStruct.hpp"
#include"NumberKeyboard.h"
#include <QMessageBox>

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

void DlgExposureTimeSet::setExposureTime(int exposureTime)
{
	ui->pbtn_exposureTimeValue->setText(QString::number(exposureTime));
}

void DlgExposureTimeSet::build_ui()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& exposureTimeSetConfig = globalStruct.dlgExposureTimeSetConfig;
	ui->pbtn_exposureTimeValue->setText(QString::number(exposureTimeSetConfig.exposureTime));
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
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	if (globalStruct.camera1) {
		globalStruct.camera1->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera1->setFrameRate(5);
	}

	if (globalStruct.camera2) {
		globalStruct.camera2->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera2->setFrameRate(5);
	}
}

void DlgExposureTimeSet::ResetCamera()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

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
}

void DlgExposureTimeSet::pbtn_exposureTimeValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto newValue = numKeyBord.getValue().toInt();
		if (newValue < 0) {
			QMessageBox::warning(this, "错误", "曝光时间范围应设置为0~300");
			return;
		}
		if (newValue > 300) {
			QMessageBox::warning(this, "错误", "曝光时间范围应设置为0~300");
			return;
		}

		auto& globalStruct = GlobalStructDataZipper::getInstance();
		ui->pbtn_exposureTimeValue->setText(QString::number(newValue));
		globalStruct.dlgExposureTimeSetConfig.exposureTime = newValue;

		globalStruct.setCameraExposureTime(1, newValue);
		globalStruct.setCameraExposureTime(2, newValue);

		globalStruct.saveDlgExposureTimeSetConfig();

		this->close();
	}
}

void DlgExposureTimeSet::pbtn_close_clicked()
{
	this->close();
}