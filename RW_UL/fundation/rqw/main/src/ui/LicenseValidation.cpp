#include "LicenseValidation.h"

//#include "hoei_GPUInfo.hpp"
//#include "ui_LicenseValidation.h"
//#include"hoei_HardwareInfo.hpp"
//
//#include"FullKeyBoard.h"
//
//LicenseValidation::LicenseValidation(QWidget* parent)
//	: QDialog(parent)
//	, ui(new Ui::LicenseValidationClass())
//{
//	ui->setupUi(this);
//
//	build_ui();
//	build_connect();
//}
//
//LicenseValidation::~LicenseValidation()
//{
//	delete ui;
//}
//
//void LicenseValidation::build_ui()
//{
//	fullKeyBoard = new FullKeyBoard(this);
//	auto hardwareInfo = rw::hoei::HardwareInfoFactory::createHardwareInfo();
//	ui->label_seriaNumber->setText(QString::fromStdString(hardwareInfo.motherBoard.UUID));
//}
//
//void LicenseValidation::build_connect()
//{
//	connect(ui->pbtn_exit, &QPushButton::clicked, this, &LicenseValidation::pbtn_exit_clicked);
//	connect(ui->pbtn_activative, &QPushButton::clicked, this, &LicenseValidation::pbtn_activative_clicked);
//	connect(ui->pbtn_ok, &QPushButton::clicked, this, &LicenseValidation::pbtn_ok_clicked);
//	connect(ui->pbtn_activationCode, &QPushButton::clicked, this, &LicenseValidation::pbtn_activationCode_clicked);
//}
//
//void LicenseValidation::pbtn_exit_clicked()
//{
//	this->reject();
//}
//
//void LicenseValidation::pbtn_activative_clicked()
//{
//}
//
//void LicenseValidation::pbtn_ok_clicked()
//{
//	this->accept();
//}
//
//void LicenseValidation::pbtn_activationCode_clicked()
//{
//	auto result = fullKeyBoard->exec();
//	if (result == QDialog::Accepted)
//	{
//		auto str = fullKeyBoard->getValue();
//		ui->pbtn_activationCode->setText(str);
//	}
//}